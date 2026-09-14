#!/usr/bin/env python3
"""
FastAPI-based Policy Server for Remote Inference (HTTP/JSON).

Supports multiple concurrent clients with batched inference.

Client sends a JSON payload containing a serialized MetaObs (numpy arrays are base64-encoded).
Server collects requests, batches them, runs inference, and returns actions as JSON.

This is an HTTP/JSON alternative to the TCP+pickle server in `deploy/comm/server.py`.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import threading
import uuid
from collections import deque
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from loguru import logger

import configs  # noqa: F401 (kept for side-effects/config registration)
from benchmark.base import MetaObs, dict2meta
from data_utils.normalize import load_normalizers
from data_utils.utils import set_seed

from ..base import BaseServer


from .codec import (
    JPEG_TYPE_NAME,
    NDARRAY_TYPE_TAG,
    NDARRAY_TYPE_NAME,
    decode_images_jpeg,
    decode_ndarray as _decode_ndarray,
    encode_ndarray as _encode_ndarray,
)


def _to_jsonable(x: Any) -> Any:
    """
    Convert arbitrary Python objects to JSON-serializable payload.
    - numpy arrays -> tagged dict with base64 buffer
    - numpy scalars -> Python scalars
    - dataclasses -> dict
    - lists/tuples/dicts -> recursively converted
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        # IMPORTANT: object arrays (e.g., MetaPolicy outputs) contain Python dicts.
        # Encoding them via `.tobytes()` would serialize memory pointers and is not portable.
        if x.dtype == object:
            return [_to_jsonable(v) for v in x.tolist()]
        return _encode_ndarray(x)
    if isinstance(x, (np.generic,)):
        return x.item()
    if hasattr(x, "__dataclass_fields__"):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # Numpy object arrays often appear in MetaPolicy outputs.
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _from_jsonable(x: Any) -> Any:
    """
    Inverse of `_to_jsonable` for data we emit/accept.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        tag = x.get(NDARRAY_TYPE_TAG)
        if tag == NDARRAY_TYPE_NAME:
            return _decode_ndarray(x)
        if tag == JPEG_TYPE_NAME:
            return decode_images_jpeg(x)
        return {k: _from_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_jsonable(v) for v in x]
    return x


def _payload_to_metaobs(meta_obs_payload: Dict[str, Any]) -> MetaObs:
    decoded = _from_jsonable(meta_obs_payload)
    if not isinstance(decoded, dict):
        raise ValueError("meta_obs must decode to a dict")
    return dict2meta(decoded, mtype="obs")


def _enrich_samples(samples: List[dict], payload: Dict[str, Any]) -> None:
    """
    Enrich samples with metadata from request payload.
    Modifies samples in-place.
    """
    reasoning = _from_jsonable(payload.get("reasoning", None)) if payload.get("reasoning", None) is not None else None
    episode_id = payload.get("episode_id", -1)
    index = payload.get("__index__", -1)
    ts = payload.get("timestamp", None)

    for s in samples:
        # NOTE: Do NOT set action/is_pad to None; if key exists, data_collator will try to
        # convert it and fail. Leave them absent so `'action' in sample` returns False.
        if reasoning is not None:
            s.setdefault("reasoning", reasoning)
        s.setdefault("episode_id", episode_id)
        s.setdefault("__index__", index)
        # `normed_mobs_to_samples` fills timestamp from mobs.timestep; keep it if already present.
        if "timestamp" not in s and ts is not None:
            s["timestamp"] = ts


def _maybe_post_process_denormalized_macts(meta_policy, macts, policy_obs):
    """After denormalize and reshape to (B, T, dim) or (B, dim); before transpose to time-major."""
    pol = meta_policy.policy
    if hasattr(pol, "post_process_action") and callable(pol.post_process_action):
        macts.action = pol.post_process_action(
            macts.action,
            policy_obs,
            meta_policy.action_normalizer,
            meta_policy.state_normalizer,
        )


def _infer_via_standard_samples(policy, mobs: MetaObs, payload: Dict[str, Any]):
    """
    Build ILStudio standard-format samples (per `.cursor/rules/data-rule.mdc`)
    and feed them into the model for inference.

    This mirrors `benchmark.base.MetaPolicy.inference`, but ensures the sample dict includes:
    - image/state/raw_lang/timestamp
    - action/is_pad placeholders (None) for inference
    - reasoning/episode_id/__index__ metadata from request payload (best-effort)
    """
    # Normalize state first (same as MetaPolicy.inference)
    normed_mobs = policy.state_normalizer.normalize_metaobs(mobs, policy.ctrl_space)
    samples = policy.normed_mobs_to_samples(normed_mobs)

    # Enrich samples with metadata
    _enrich_samples(samples, payload)

    # Convert samples to model input
    policy_obs = policy.meta2obs(samples)

    action_chunk = policy.policy.select_action(policy_obs)

    # Convert to MetaAction and denormalize action (same as MetaPolicy.inference)
    macts = policy.act2meta(action_chunk, ctrl_space=policy.ctrl_space, ctrl_type=policy.ctrl_type)
    macts = policy.action_normalizer.denormalize_metaact(macts)

    _maybe_post_process_denormalized_macts(policy, macts, policy_obs)

    macts.action = macts.action[np.newaxis, :] if len(macts.action.shape)<3 else macts.action.swapaxes(0,1)

    from benchmark.base import MetaAction

    mact_list = [
        np.array(
            [asdict(MetaAction(action=aii, ctrl_type=macts.ctrl_type, ctrl_space=macts.ctrl_space)) for aii in ai],
            dtype=object,
        )
        for ai in macts.action
    ]

    return mact_list


class BatchedInferenceManager:
    """
    Manages batched inference for multiple concurrent HTTP requests.
    
    Collects requests over a short time window (batch_wait_ms), batches them,
    runs inference, and distributes results back to waiting requests.
    
    Key feature: Per-client deduplication
    - Each client is identified by `client_id` in the request payload
    - When multiple requests from the same client arrive before processing,
      only the LATEST request is processed (others receive the same result)
    - This prevents stale obs from being processed and saves compute
    """

    def __init__(self, policy, batch_wait_ms: float = 1.0):
        """
        Args:
            policy: The MetaPolicy to use for inference.
            batch_wait_ms: Time to wait for collecting requests before batching (ms).
        """
        self.policy = policy
        self.batch_wait_ms = batch_wait_ms
        
        # Per-client latest request: client_id -> (mobs, payload, result_event, result_container, seq_num)
        # Using dict ensures only latest request per client is kept
        self._client_requests: Dict[str, Tuple] = {}
        # List of (result_event, result_container) for superseded requests (they share result with latest)
        self._superseded_requests: Dict[str, List[Tuple]] = {}
        self._queue_lock = threading.Lock()
        
        # Sequence number for ordering requests
        self._seq_counter = 0
        
        # Inference thread
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        
        self._request_count = 0
        self._batch_count = 0
        self._superseded_count = 0  # Count of requests that were superseded

    def submit_request(
        self, 
        mobs: MetaObs, 
        payload: Dict[str, Any], 
        client_id: Optional[str] = None,
        timeout: float = 3600.0,
    ) -> list:
        """
        Submit a request and wait for the result.
        
        Args:
            mobs: MetaObs to infer.
            payload: Original request payload (for metadata).
            client_id: Unique client identifier. If provided, only the latest request
                       from this client will be processed; earlier requests will receive
                       the same result as the latest one.
            timeout: Maximum time to wait for result (seconds).
            
        Returns:
            mact_list: List of action arrays.
            
        Raises:
            TimeoutError: If inference takes too long.
            RuntimeError: If inference fails.
        """
        result_event = threading.Event()
        result_container = {"result": None, "error": None}
        
        # Use client_id from parameter or payload, fallback to unique request_id
        if client_id is None:
            client_id = payload.get("client_id")
        if client_id is None:
            # No client_id means no deduplication for this request
            client_id = f"__anon_{uuid.uuid4()}"
        
        with self._queue_lock:
            self._seq_counter += 1
            seq_num = self._seq_counter
            
            # Check if there's already a pending request from this client
            if client_id in self._client_requests:
                # Move the old request to superseded list (it will share result with latest)
                old_req = self._client_requests[client_id]
                old_event, old_container = old_req[2], old_req[3]
                
                if client_id not in self._superseded_requests:
                    self._superseded_requests[client_id] = []
                self._superseded_requests[client_id].append((old_event, old_container))
                self._superseded_count += 1
            
            # Store as the latest request for this client
            self._client_requests[client_id] = (mobs, payload, result_event, result_container, seq_num)
        
        # Wait for result
        if not result_event.wait(timeout=timeout):
            # Cleanup on timeout
            with self._queue_lock:
                if client_id in self._client_requests:
                    req = self._client_requests[client_id]
                    if req[4] == seq_num:  # Still our request
                        del self._client_requests[client_id]
            raise TimeoutError(f"Inference timeout after {timeout}s")
        
        if result_container["error"] is not None:
            raise RuntimeError(f"Inference failed: {result_container['error']}")
        
        return result_container["result"]

    def _inference_loop(self) -> None:
        """Background thread that processes batched requests."""
        with torch.no_grad():
            while self._running:
                try:
                    # Collect requests
                    batch_data = self._collect_batch()
                    
                    if not batch_data:
                        time.sleep(0.0001)  # 0.1ms idle sleep
                        continue
                    
                    # Run batched inference
                    self._process_batch(batch_data)
                    
                except Exception as e:
                    logger.error(f"Inference loop error: {e}")
                    import traceback
                    traceback.print_exc()

    def _collect_batch(self) -> List[Tuple]:
        """
        Collect pending requests into a batch.
        
        Returns list of (client_id, mobs, payload, result_event, result_container, superseded_list)
        """
        batch = []
        
        with self._queue_lock:
            if not self._client_requests:
                return batch
        
        # Wait a bit to collect more requests (allows newer requests to supersede older ones)
        if self.batch_wait_ms > 0:
            time.sleep(self.batch_wait_ms / 1000.0)
        
        with self._queue_lock:
            if not self._client_requests:
                return batch
            
            # Take all current requests
            for client_id, req in self._client_requests.items():
                mobs, payload, event, container, seq_num = req
                superseded = self._superseded_requests.pop(client_id, [])
                batch.append((client_id, mobs, payload, event, container, superseded))
            
            self._client_requests.clear()
        
        return batch

    def _process_batch(self, batch: List[Tuple]) -> None:
        """Process a batch of requests."""
        if not batch:
            return
        
        n_requests = len(batch)
        client_ids = [b[0] for b in batch]
        mobs_list = [b[1] for b in batch]
        payloads = [b[2] for b in batch]
        events = [b[3] for b in batch]
        containers = [b[4] for b in batch]
        superseded_lists = [b[5] for b in batch]
        
        try:
            if n_requests == 1:
                # Single request - no batching needed
                mact_list = _infer_via_standard_samples(self.policy, mobs_list[0], payloads[0])
                containers[0]["result"] = mact_list
            else:
                # Multiple requests - batch inference
                results = self._batch_inference(mobs_list, payloads)
                
                for i, result in enumerate(results):
                    containers[i]["result"] = result
            
            self._request_count += n_requests
            self._batch_count += 1
            
            if self._batch_count % 100 == 0:
                logger.debug(f"  Processed {self._request_count} requests in {self._batch_count} batches, "
                           f"avg batch size: {self._request_count / self._batch_count:.1f}, "
                           f"superseded: {self._superseded_count}")
                
        except Exception as e:
            # Set error for all requests in batch
            error_msg = str(e)
            for container in containers:
                container["error"] = error_msg
        
        finally:
            # Signal all waiting requests (including superseded ones)
            for i, event in enumerate(events):
                # Copy result to superseded requests
                for sup_event, sup_container in superseded_lists[i]:
                    sup_container["result"] = containers[i]["result"]
                    sup_container["error"] = containers[i]["error"]
                    sup_event.set()
                
                event.set()

    def _batch_inference(self, mobs_list: List[MetaObs], payloads: List[Dict]) -> List[list]:
        """
        Run batched inference on multiple MetaObs.
        
        Args:
            mobs_list: List of MetaObs to infer.
            payloads: List of request payloads (for metadata).
            
        Returns:
            List of mact_list, one per input MetaObs.
        """
        n_clients = len(mobs_list)
        
        # Batch the MetaObs
        batched_mobs = self._batch_metaobs(mobs_list)
        
        # Normalize state
        normed_mobs = self.policy.state_normalizer.normalize_metaobs(batched_mobs, self.policy.ctrl_space)
        samples = self.policy.normed_mobs_to_samples(normed_mobs)
        
        # Enrich samples with metadata from first payload (best-effort)
        _enrich_samples(samples, payloads[0])
        
        # Convert samples to model input
        policy_obs = self.policy.meta2obs(samples)
        
        action_chunk = self.policy.policy.select_action(policy_obs)
        
        # Convert to MetaAction and denormalize action
        macts = self.policy.act2meta(action_chunk, ctrl_space=self.policy.ctrl_space, ctrl_type=self.policy.ctrl_type)
        
        macts = self.policy.action_normalizer.denormalize_metaact(macts)

        _maybe_post_process_denormalized_macts(self.policy, macts, policy_obs)
        
        macts.action = macts.action[np.newaxis, :] if len(macts.action.shape)<3 else macts.action.swapaxes(0,1)

        from benchmark.base import MetaAction
        
        # Build full mact_list (batched)
        mact_list = [
            np.array(
                [asdict(MetaAction(action=aii, ctrl_type=macts.ctrl_type, ctrl_space=macts.ctrl_space)) for aii in ai],
                dtype=object,
            )
            for ai in macts.action
        ]
        
        # No truncation — chunk management is delegated to client-side action_manager
        
        # Unbatch results
        return self._unbatch_mact_list(mact_list, n_clients)

    def _batch_metaobs(self, mobs_list: List[MetaObs]) -> MetaObs:
        """Batch multiple MetaObs into one."""
        if len(mobs_list) == 1:
            return mobs_list[0]
        
        # Helper to safely concatenate arrays
        def safe_concat(attr_name):
            arrays = [getattr(m, attr_name) for m in mobs_list if getattr(m, attr_name, None) is not None]
            if not arrays:
                return None
            return np.concatenate(arrays, axis=0)
        
        # Stack all fields along batch dimension
        batched = MetaObs(
            state=safe_concat('state'),
            state_ee=safe_concat('state_ee'),
            state_joint=safe_concat('state_joint'),
            state_obj=safe_concat('state_obj'),
            image=safe_concat('image'),
            depth=safe_concat('depth'),
            pc=safe_concat('pc'),
            timestep=safe_concat('timestep'),
            raw_lang=mobs_list[0].raw_lang if mobs_list[0].raw_lang else '',
        )
        return batched

    def _unbatch_mact_list(self, mact_list: list, n_clients: int) -> List[list]:
        """
        Unbatch mact_list back to per-client results.
        
        Args:
            mact_list: Batched action list from inference
            n_clients: Number of clients in the batch
            
        Returns:
            List of mact_list, one per client.
        """
        if n_clients == 1:
            return [mact_list]
        
        # mact_list is List[np.ndarray(dtype=object)] where each array has batch_size elements
        results = [[] for _ in range(n_clients)]
        
        for step_arr in mact_list:
            if isinstance(step_arr, np.ndarray) and len(step_arr) == n_clients:
                # Split by client
                for i in range(n_clients):
                    results[i].append(np.array([step_arr[i]], dtype=object))
            else:
                # Can't unbatch properly, give same result to all
                for i in range(n_clients):
                    results[i].append(step_arr)
        
        return results

    def stop(self) -> None:
        """Stop the inference thread."""
        self._running = False
        if self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)


def create_app(policy, batch_wait_ms: float = 1.0) -> FastAPI:
    """
    Create FastAPI app with batched inference support.
    
    Args:
        policy: The MetaPolicy to serve.
        batch_wait_ms: Time to wait for collecting requests before batching (ms).
    """
    app = FastAPI(title="ILStudio Policy Server (FastAPI)")
    app.state.policy = policy
    app.state.batch_manager = BatchedInferenceManager(policy, batch_wait_ms=batch_wait_ms)

    @app.get("/health")
    async def health():
        manager = app.state.batch_manager
        return {
            "status": "ok",
            "requests_processed": manager._request_count,
            "batches_processed": manager._batch_count,
            "superseded_requests": manager._superseded_count,
            # Advertised to FastAPIPolicyClient (image_encoding=auto → jpeg).
            "image_codecs": [JPEG_TYPE_NAME, "jpeg", NDARRAY_TYPE_NAME],
        }

    @app.post("/inference")
    async def inference(payload: Dict[str, Any] = Body(...)):
        """
        Expected payload shape:
        {
          "meta_obs": { ... serialized MetaObs ... },
          "client_id": optional str (for request deduplication - only latest request per client is processed),
          "episode_id": optional int,
          "__index__": optional int,
          "reasoning": optional any,
          "timestamp": optional float|int (if you want to override MetaObs.timestep)
        }
        
        Note on client_id:
        - If provided, when multiple requests from the same client arrive before processing,
          only the LATEST observation will be used for inference.
        - Earlier requests will receive the same action result as the latest one.
        - This prevents stale observations from consuming compute and ensures actions match current state.
        """
        try:
            if "meta_obs" not in payload:
                raise ValueError("Missing field: meta_obs")

            mobs = _payload_to_metaobs(payload["meta_obs"])

            # Optional overrides / metadata (best-effort; model may ignore)
            if "timestamp" in payload and payload["timestamp"] is not None:
                mobs.timestep = _from_jsonable(payload["timestamp"])

            # Extract client_id for deduplication
            client_id = payload.get("client_id")

            # Submit to batch manager and wait for result
            mact_list = app.state.batch_manager.submit_request(mobs, payload, client_id=client_id)

            # Return actions (JSON-encoded)
            return {"mact_list": _to_jsonable(mact_list)}
        except TimeoutError as e:
            logger.error(f"FastAPI inference timeout: {e}")
            raise HTTPException(status_code=504, detail=str(e)) from e
        except Exception as e:
            logger.exception("FastAPI inference failed")
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on server shutdown."""
        if hasattr(app.state, "batch_manager"):
            app.state.batch_manager.stop()

    return app


class FastAPIPolicyServer(BaseServer):
    """
    HTTP/JSON policy server using FastAPI + uvicorn with batched inference.

    Inherits from BaseServer for a unified interface.

    Args:
        policy: The MetaPolicy to serve.
        host: Bind address (e.g., "0.0.0.0").
        port: Port number.
        ssl_keyfile: Path to SSL private key file (for HTTPS).
        ssl_certfile: Path to SSL certificate file (for HTTPS).
        batch_wait_ms: Time to wait for collecting requests before batching (ms).
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        batch_wait_ms: float = 1.0,
    ):
        self.policy = policy
        self.host = host
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.batch_wait_ms = batch_wait_ms
        self._app: Optional[FastAPI] = None
        self._server = None

    def start(self) -> None:
        """Start the server (blocking)."""
        self._app = create_app(self.policy, batch_wait_ms=self.batch_wait_ms)
        scheme = "https" if self.ssl_certfile else "http"
        logger.info(f"🚀 FastAPI Policy Server started on {scheme}://{self.host}:{self.port}")
        logger.info(f"   Batch wait: {self.batch_wait_ms}ms")
        uvicorn.run(
            self._app,
            host=self.host,
            port=self.port,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
        )

    def stop(self) -> None:
        """Stop the server gracefully (no-op for uvicorn.run blocking mode)."""
        if self._app and hasattr(self._app.state, "batch_manager"):
            self._app.state.batch_manager.stop()
        logger.info("✓ FastAPI Policy Server stopped")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Start ILStudio FastAPI policy server",
        epilog=(
            "For HTTPS, set environment variables:\n"
            "  export ILSTD_SSL_KEYFILE=/path/to/key.pem\n"
            "  export ILSTD_SSL_CERTFILE=/path/to/cert.pem"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("-p", "--port", type=int, default=8000)
    p.add_argument("--batch_wait_ms", type=float, default=1.0,
                   help="Time to wait for collecting requests before batching (ms)")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="ckpt/act_sim_transfer_cube_scripted_zscore_example",
    )
    p.add_argument("--dataset_id", type=str, default="")
    p.add_argument("--chunk_size", type=int, default=-1)
    p.add_argument(
        "--https",
        action="store_true",
        help="Enable HTTPS (requires ILSTD_SSL_KEYFILE and ILSTD_SSL_CERTFILE env vars)",
    )
    return p.parse_args()


def _signal_handler(signum, frame):
    logger.info("⏸ Received interrupt signal, shutting down...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    set_seed(0)
    args = _parse_args()
    args.is_training = False

    logger.info("=" * 60)
    logger.info("🚀 FastAPI Policy Server Startup")
    logger.info("=" * 60)
    logger.info(f"   Model path: {args.model_name_or_path}")
    logger.info(f"   Dataset ID: {args.dataset_id if args.dataset_id else '(first dataset)'}")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Batch wait: {args.batch_wait_ms}ms")

    # Load normalizers
    normalizers, ctrl_space, ctrl_type = load_normalizers(args)
    args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type

    # Load policy from checkpoint
    from policy.direct_loader import load_model_from_checkpoint
    from benchmark.base import MetaPolicy

    model_components = load_model_from_checkpoint(args.model_name_or_path, args)
    model = model_components["model"]
    model.eval()

    policy = MetaPolicy(
        policy=model,
        action_normalizer=normalizers["action"],
        state_normalizer=normalizers["state"],
        ctrl_space=ctrl_space,
        ctrl_type=ctrl_type,
    )

    # Get SSL config from environment if HTTPS is requested
    ssl_keyfile = None
    ssl_certfile = None
    if getattr(args, "https", False):
        ssl_keyfile = os.environ.get("ILSTD_SSL_KEYFILE")
        ssl_certfile = os.environ.get("ILSTD_SSL_CERTFILE")
        if not ssl_keyfile or not ssl_certfile:
            logger.error(
                "HTTPS requires SSL certificates. Please set the following environment variables:\n"
                "\n"
                "    export ILSTD_SSL_KEYFILE=/path/to/your/key.pem\n"
                "    export ILSTD_SSL_CERTFILE=/path/to/your/cert.pem\n"
                "\n"
                "To generate a self-signed certificate for testing:\n"
                "\n"
                "    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj \"/CN=localhost\"\n"
                "    export ILSTD_SSL_KEYFILE=key.pem\n"
                "    export ILSTD_SSL_CERTFILE=cert.pem\n"
            )
            sys.exit(1)

    server = FastAPIPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        batch_wait_ms=args.batch_wait_ms,
    )
    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
