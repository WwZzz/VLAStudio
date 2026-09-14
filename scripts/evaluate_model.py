"""
Evaluate a policy checkpoint on the training split: forward training loss (no backward)
and action metrics (MSE / MAE) from select_action vs ground-truth.

- Policy YAML is inferred from checkpoint ``policy_metadata.json`` (no ``-p``).
- Task is ``-t`` only; after load, dataset ``chunk_size`` is reset from the loaded model's config (not policy yaml merge).
- Training YAML is fixed internally (``configs/training/default.yaml``) for dataloader / TrainingArguments only, not a CLI flag.
- ``output_dir`` for TrainingArguments is set automatically (checkpoint directory if ``-m`` is local, else ``ckpt/eval_scratch``); no flag.
- Each task sub-dataset is wrapped with normalizers loaded from the checkpoint's
  ``normalize.json`` and per-``dataset_id`` ``*_stats_*.pkl`` files. Sub-datasets
  missing from the checkpoint metadata or stats on disk are warned and skipped.

Remote addresses (host:port, http://, shm://) are not supported here.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

from vlastudio import configs  # noqa: F401 — suppress TF logs etc., keep first side effects
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vlastudio.data_utils.data_loader import WrappedDataset, get_dataloader, is_iter_data, is_map_data
from vlastudio.data_utils.dataset_wrappers import wrap_dataset_with_normalizers
from vlastudio.data_utils.normalize import load_normalizer_from_meta
from vlastudio.data_utils.utils import (
    _apply_transforms_to_datasets,
    _create_vqa_dataset_from_config,
    _maybe_assign_weights_to_datasets,
    _parse_datasets_config,
    _train_val_split_datasets,
    _wrap_vqa_datasets,
    set_seed,
)
from vlastudio.policy.policy_loader import (
    get_policy_data_collator,
    get_policy_data_processor,
    load_policy_model_for_training,
)
from vlastudio.policy.utils import is_server_address
from train import load_all_configs

from vlastudio.configs.loader import ConfigLoader

# Fixed for eval: only supplies TrainingArguments / dataloader defaults (not user-facing).
_EVAL_TRAINING_CONFIG = "default"


def apply_checkpoint_chunk_size_to_task(task_config: dict, chunk_size: int | None) -> None:
    """After merge_all_parameters (policy yaml may force chunk_size), match the loaded ckpt weights."""
    if chunk_size is None:
        return
    for ds_cfg in task_config.get("datasets", []):
        if not isinstance(ds_cfg, dict):
            continue
        if ConfigLoader._is_merged_dataset_config(ds_cfg):
            for _mid, sub_configs in ds_cfg.items():
                if not isinstance(sub_configs, list):
                    continue
                for sub in sub_configs:
                    if not isinstance(sub, dict) or "args" not in sub:
                        continue
                    if "chunk_size" not in sub["args"]:
                        continue
                    old = sub["args"]["chunk_size"]
                    if old != chunk_size:
                        logger.info(
                            f"Eval: dataset chunk_size {old} -> {chunk_size} (checkpoint model.config)"
                        )
                        sub["args"]["chunk_size"] = chunk_size
        elif "args" in ds_cfg and "chunk_size" in ds_cfg["args"]:
            old = ds_cfg["args"]["chunk_size"]
            if old != chunk_size:
                logger.info(
                    f"Eval: dataset chunk_size {old} -> {chunk_size} (checkpoint model.config)"
                )
                ds_cfg["args"]["chunk_size"] = chunk_size


def default_eval_output_dir(model_path: str) -> str:
    """load_all_configs needs hyper_args.output_dir; use local checkpoint dir when possible."""
    p = Path(model_path)
    if p.is_dir():
        return str(p.resolve())
    scratch = _ROOT / "ckpt" / "eval_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return str(scratch.resolve())


def parse_param():
    parser = argparse.ArgumentParser(
        description="Eval loss + select_action vs GT on the training dataset (batched, no grad)."
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help="Checkpoint path, HF model id, or hub snapshot (same as training from_pretrained).",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="sim_transfer_cube_scripted",
        help="Task config (name under configs/task or path to yaml)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for the model")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Override batch size; default: per_device_train_batch_size from built-in eval profile",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Stop after this many batches (debug / smoke test)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, write metrics dict to this path",
    )
    args, unknown = parser.parse_known_args()
    args.unknown_args = unknown
    return args


def resolve_checkpoint_root(model_name_or_path: str) -> Path:
    p = Path(model_name_or_path).expanduser()
    if p.name.startswith("checkpoint-"):
        return p.parent
    return p


def infer_policy_key_from_checkpoint(ckpt_root: Path) -> str:
    meta_path = ckpt_root / "policy_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Cannot infer policy: missing {meta_path}. "
            "Expected policy_metadata.json next to normalize.json (training output root or HF snapshot)."
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)
    name = meta.get("policy_name")
    if not name and meta.get("policy_module"):
        name = str(meta["policy_module"]).split(".")[-1]
    if not name:
        raise ValueError(f"policy_metadata.json has no policy_name / policy_module: {meta_path}")
    name = str(name)
    policy_dir = _ROOT / "configs" / "policy"
    for cand in (name, name.lower()):
        if (policy_dir / f"{cand}.yaml").is_file():
            return cand
    raise FileNotFoundError(
        f"No policy yaml for '{name}' under {policy_dir} (tried exact and lowercase)."
    )


def load_checkpoint_normalize_meta(ckpt_root: Path) -> dict:
    norm_path = ckpt_root / "normalize.json"
    if not norm_path.is_file():
        raise FileNotFoundError(
            f"Missing {norm_path}. Cannot align per-dataset normalizers with the checkpoint."
        )
    with open(norm_path, "r") as f:
        return json.load(f)


def _dataset_meta_for_id(norm_meta: dict, dataset_id: str) -> dict | None:
    for ds in norm_meta.get("datasets", []):
        if ds.get("dataset_id") == dataset_id:
            return ds
    return None


def checkpoint_has_dataset_stats(ckpt_root: Path, norm_meta: dict, dataset_id: str) -> tuple[bool, str]:
    if _dataset_meta_for_id(norm_meta, dataset_id) is None:
        return False, f"dataset_id '{dataset_id}' not listed in checkpoint normalize.json"
    st = norm_meta.get("state", {})
    ac = norm_meta.get("action", {})
    if dataset_id not in st or dataset_id not in ac:
        return (
            False,
            f"dataset_id '{dataset_id}' missing from normalize.json state/action entries",
        )
    ds_entry = _dataset_meta_for_id(norm_meta, dataset_id)
    ctrl_space = ds_entry.get("ctrl_space", "ee")
    ctrl_type = ds_entry.get("ctrl_type", "delta")
    stats_name = f"{dataset_id}_stats_{ctrl_space}_{ctrl_type}.pkl"
    stats_path = ckpt_root / stats_name
    if not stats_path.is_file():
        return False, f"missing stats file in checkpoint: {stats_name}"
    return True, ""


def load_train_data_with_checkpoint_normalizers(
    args, task_config: dict, ckpt_root: Path, norm_meta: dict
):
    """
    Like load_data train path, but normalizers are loaded from the checkpoint per dataset_id.
    Sub-datasets without matching checkpoint stats are skipped (with warning).
    """
    if "datasets" not in task_config and "vqa" not in task_config:
        raise ValueError("Task config has no 'datasets' or 'vqa' section")
    evaluated_ids: list[str] = []
    skipped: list[dict] = []
    wrapped_robot: list = []
    datasets: list = []

    if "datasets" in task_config:
        raw_datasets, _flattened, _merge_info = _parse_datasets_config(
            task_config["datasets"], args
        )
        for ds in raw_datasets:
            did = getattr(ds, "dataset_id", None)
            if not did:
                msg = "dataset has no dataset_id"
                logger.warning(f"Skipping a dataset: {msg}")
                skipped.append({"dataset_id": None, "reason": msg})
                continue
            ok, reason = checkpoint_has_dataset_stats(ckpt_root, norm_meta, did)
            if not ok:
                logger.warning(f"Skipping dataset '{did}': {reason}")
                skipped.append({"dataset_id": did, "reason": reason})
                continue
            try:
                nz = load_normalizer_from_meta(
                    norm_meta, src_dir=str(ckpt_root), dataset_id=did
                )
            except Exception as e:
                logger.warning(f"Skipping dataset '{did}': could not load normalizers ({e})")
                skipped.append({"dataset_id": did, "reason": f"load_normalizer_from_meta: {e}"})
                continue
            wrapped_robot.append(
                wrap_dataset_with_normalizers(
                    ds,
                    action_normalizers={did: nz["action"]},
                    state_normalizers={did: nz["state"]},
                    dataset_name=did,
                )
            )
            evaluated_ids.append(did)

        datasets = _apply_transforms_to_datasets(wrapped_robot, args, task_config)

    if "vqa" in task_config:
        vqa_cfgs = task_config["vqa"]
        vqa_raw = [_create_vqa_dataset_from_config(c, args) for c in vqa_cfgs]
        vqa_t = _apply_transforms_to_datasets(vqa_raw, args, task_config)
        vqa_wrapped = _wrap_vqa_datasets(vqa_t, args, task_config)
        for cfg, vds in zip(vqa_cfgs, vqa_wrapped):
            vid = cfg.get("name")
            if not vid:
                continue
            ok, reason = checkpoint_has_dataset_stats(ckpt_root, norm_meta, vid)
            if not ok:
                logger.warning(f"Skipping VQA dataset '{vid}': {reason}")
                skipped.append({"dataset_id": vid, "reason": reason})
                continue
            try:
                nz = load_normalizer_from_meta(
                    norm_meta, src_dir=str(ckpt_root), dataset_id=vid
                )
            except Exception as e:
                logger.warning(f"Skipping VQA dataset '{vid}': {e}")
                skipped.append({"dataset_id": vid, "reason": str(e)})
                continue
            datasets.append(
                wrap_dataset_with_normalizers(
                    vds,
                    action_normalizers={vid: nz["action"]},
                    state_normalizers={vid: nz["state"]},
                    dataset_name=vid,
                )
            )
            evaluated_ids.append(vid)

    if not datasets:
        raise RuntimeError(
            "No datasets left to evaluate after checkpoint normalizer filtering. "
            "Check task dataset_id names match normalize.json and stats pickles under the checkpoint root."
        )

    train_data, _eval_data = _train_val_split_datasets(datasets, args)
    train_data = _maybe_assign_weights_to_datasets(train_data, task_config)
    return train_data, evaluated_ids, skipped


def filter_forward_kwargs(model: torch.nn.Module, batch: dict) -> dict:
    try:
        sig = inspect.signature(model.forward)
        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return dict(batch)
        names = {p.name for p in params if p.name != "self"}
        return {k: v for k, v in batch.items() if k in names}
    except (TypeError, ValueError):
        return dict(batch)


def extract_loss_tensor(outputs) -> torch.Tensor | None:
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        if "loss" in outputs and torch.is_tensor(outputs["loss"]):
            return outputs["loss"]
        return None
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    return None


def compute_forward_loss(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor | None:
    """
    Run the training forward path to obtain scalar loss, without backward.
    Some policies (e.g. flow-matching VLA) only expose the loss branch when model.training is True.
    """
    was_training = model.training
    model.train()
    try:
        b = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                b[k] = v.to(device, non_blocking=True)
            else:
                b[k] = v
        fwd_kw = filter_forward_kwargs(model, b)
        with torch.inference_mode():
            outputs = model(**fwd_kw)
        loss = extract_loss_tensor(outputs)
        return loss
    except Exception as e:
        logger.warning(f"Forward loss skipped for a batch: {e}")
        return None
    finally:
        if not was_training:
            model.eval()


def _align_pred_gt_shapes(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pred.shape == gt.shape:
        return pred, gt
    ps, gs = pred.shape, gt.shape
    if len(ps) == 3 and len(gs) == 3:
        min_t = min(ps[1], gs[1])
        return pred[:, :min_t, :], gt[:, :min_t, :]
    if len(ps) == 3 and len(gs) == 2:
        return pred[:, 0, :], gt
    if len(ps) == 2 and len(gs) == 3:
        return pred, gt[:, 0, :]
    if len(ps) == 3 and ps[1] == 1:
        return pred.squeeze(1), gt
    if len(gs) == 3 and gs[1] == 1:
        return pred, gt.squeeze(1)
    pred_f = pred.flatten()
    gt_f = gt.flatten()
    if pred_f.shape == gt_f.shape:
        return pred_f, gt_f
    return None


def masked_mean_mse_mae(
    pred: torch.Tensor, gt: torch.Tensor, is_pad: torch.Tensor | None
) -> tuple[float, float, int] | None:
    all_mse = F.mse_loss(pred, gt, reduction="none")
    all_mae = F.l1_loss(pred, gt, reduction="none")
    if is_pad is not None:
        if is_pad.dim() == 2 and pred.dim() == 3:
            mask = ~is_pad.unsqueeze(-1)
            mask = mask.expand_as(all_mse)
            num_valid = int(mask.sum().item())
            if num_valid <= 0:
                return None
            mse = (all_mse * mask).sum() / num_valid
            mae = (all_mae * mask).sum() / num_valid
            return mse.item(), mae.item(), num_valid
        if is_pad.dim() == 1:
            mask = ~is_pad
            if pred.dim() == 3:
                mask = mask.unsqueeze(1).unsqueeze(2).expand_as(pred)
            elif pred.dim() == 2:
                mask = mask.unsqueeze(1).expand_as(pred)
            else:
                mse = all_mse.mean()
                mae = all_mae.mean()
                return mse.item(), mae.item(), int(pred.numel())
            num_valid = int(mask.sum().item())
            if num_valid <= 0:
                return None
            mse = (all_mse * mask).sum() / num_valid
            mae = (all_mae * mask).sum() / num_valid
            return mse.item(), mae.item(), num_valid
    mse = all_mse.mean()
    mae = all_mae.mean()
    return mse.item(), mae.item(), int(pred.numel())


def build_train_loader(train_data, data_processor, data_collator, training_args, eval_batch_size: int):
    if is_map_data(train_data):
        wrapped = WrappedDataset(train_data, data_processor)
        pin_memory = getattr(training_args, "dataloader_pin_memory", False)
        num_workers = getattr(training_args, "dataloader_num_workers", 0)
        from torch.utils.data import DataLoader

        return DataLoader(
            wrapped,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if is_iter_data(train_data):
        logger.warning(
            "Iterable / RLDS-style training data: using get_dataloader(train, ...) "
            "(ordering/shuffle follows training loader; not a strict sequential epoch)."
        )
        training_args.per_device_train_batch_size = eval_batch_size
        train_loader, _ = get_dataloader(train_data, None, data_processor, data_collator, training_args)
        return train_loader
    raise TypeError(f"Unsupported dataset type for batched eval: {type(train_data)}")


def main():
    args = parse_param()
    args.training_config = _EVAL_TRAINING_CONFIG
    if is_server_address(args.model_name_or_path):
        logger.error(
            "model_name_or_path looks like a remote/IPC server. "
            "This script evaluates on dataset tensors in-process; use a local checkpoint "
            "(path or HF id) or run a policy server only for sim/real rollouts."
        )
        sys.exit(1)

    ckpt_root = resolve_checkpoint_root(args.model_name_or_path)
    try:
        args.policy = infer_policy_key_from_checkpoint(ckpt_root)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"Inferred policy config key from checkpoint: {args.policy}")

    try:
        norm_meta = load_checkpoint_normalize_meta(ckpt_root)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    args.output_dir = default_eval_output_dir(args.model_name_or_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.is_training = False
    task_config, _policy_cfg, training_args, config_paths = load_all_configs(args)
    seed = getattr(training_args, "seed", 0)
    set_seed(seed)

    eval_bs = args.eval_batch_size
    if eval_bs is None:
        eval_bs = getattr(training_args, "per_device_train_batch_size", 8)
    logger.info(f"Eval batch size: {eval_bs}")

    model_components = load_policy_model_for_training(
        config_paths["policy"], args, task_config
    )
    model = model_components["model"]
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    apply_checkpoint_chunk_size_to_task(
        task_config, getattr(model.config, "chunk_size", None)
    )

    train_data, evaluated_dataset_ids, skipped_datasets = (
        load_train_data_with_checkpoint_normalizers(args, task_config, ckpt_root, norm_meta)
    )
    logger.info(f"Evaluating on dataset_id(s) with checkpoint stats: {evaluated_dataset_ids}")
    data_processor = get_policy_data_processor(
        config_paths["policy"], args, model_components
    )
    data_collator = get_policy_data_collator(config_paths["policy"], args, model_components)
    loader = build_train_loader(
        train_data, data_processor, data_collator, training_args, eval_bs
    )

    try:
        n_batches = len(loader)
    except TypeError:
        n_batches = None

    sum_loss = 0.0
    count_loss_batches = 0
    total_mse = 0.0
    total_mae = 0.0
    total_action_elems = 0
    skipped_action_batches = 0

    if not hasattr(model, "select_action"):
        logger.warning("Model has no select_action; skipping action MSE/MAE.")

    def _running_postfix() -> dict:
        d: dict[str, str] = {}
        if count_loss_batches:
            d["loss"] = f"{sum_loss / count_loss_batches:.4f}"
        if total_action_elems:
            mse_r = total_mse / total_action_elems
            mae_r = total_mae / total_action_elems
            d["mse"] = f"{mse_r:.6f}"
            d["mae"] = f"{mae_r:.6f}"
            d["rmse"] = f"{float(np.sqrt(mse_r)):.6f}"
        if skipped_action_batches:
            d["skip"] = str(skipped_action_batches)
        return d

    pbar = tqdm(
        loader,
        total=n_batches,
        desc="train set",
        dynamic_ncols=True,
        mininterval=0.15,
    )

    for bi, batch in enumerate(pbar):
        if args.max_batches is not None and bi >= args.max_batches:
            break

        loss_t = compute_forward_loss(model, batch, device)
        if loss_t is not None:
            sum_loss += float(loss_t.detach().cpu())
            count_loss_batches += 1

        if hasattr(model, "select_action"):
            batch_obs = {k: v for k, v in batch.items() if k not in ("actions", "action", "is_pad")}
            gt_actions = batch.get("actions")
            if gt_actions is None:
                gt_actions = batch.get("action")
            if gt_actions is None:
                skipped_action_batches += 1
            else:
                gt_actions = gt_actions.to(device, non_blocking=True)
                if hasattr(model, "config") and hasattr(model.config, "action_dim"):
                    adim = model.config.action_dim
                    if gt_actions.shape[-1] > adim:
                        gt_actions = gt_actions[..., :adim]
                is_pad = batch.get("is_pad")
                if is_pad is not None:
                    is_pad = is_pad.to(device, non_blocking=True)
                try:
                    pred_actions = model.select_action(batch_obs)
                except Exception as e:
                    logger.debug(f"select_action failed: {e}")
                    skipped_action_batches += 1
                    pred_actions = None
                if pred_actions is not None:
                    if isinstance(pred_actions, np.ndarray):
                        pred_actions = torch.from_numpy(pred_actions).to(device)
                    elif not isinstance(pred_actions, torch.Tensor):
                        pred_actions = torch.tensor(pred_actions, device=device)
                    pred_actions = pred_actions.to(dtype=gt_actions.dtype)
                    aligned = _align_pred_gt_shapes(pred_actions, gt_actions)
                    if aligned is None:
                        skipped_action_batches += 1
                    else:
                        pred_actions, gt_actions = aligned
                        if pred_actions.shape != gt_actions.shape:
                            skipped_action_batches += 1
                        else:
                            stats = masked_mean_mse_mae(pred_actions, gt_actions, is_pad)
                            if stats is None:
                                skipped_action_batches += 1
                            else:
                                mse_b, mae_b, n_el = stats
                                total_mse += mse_b * n_el
                                total_mae += mae_b * n_el
                                total_action_elems += n_el

        pbar.set_postfix(_running_postfix(), refresh=True)

    pbar.close()

    metrics = {
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_root": str(ckpt_root),
        "policy_config": config_paths["policy"],
        "task_config": config_paths["task"],
        "evaluated_dataset_ids": evaluated_dataset_ids,
        "skipped_datasets": skipped_datasets,
        "eval_batch_size": eval_bs,
        "batches_loss": count_loss_batches,
        "mean_forward_loss": (sum_loss / count_loss_batches) if count_loss_batches else None,
        "action_mse": (total_mse / total_action_elems) if total_action_elems else None,
        "action_mae": (total_mae / total_action_elems) if total_action_elems else None,
        "action_rmse": (
            float(np.sqrt(total_mse / total_action_elems)) if total_action_elems else None
        ),
        "action_metric_elements": total_action_elems,
        "skipped_action_batches": skipped_action_batches,
    }

    logger.info("=== Metrics (training split) ===")
    logger.info(json.dumps(metrics, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

