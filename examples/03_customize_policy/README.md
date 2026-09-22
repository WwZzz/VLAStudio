# Custom Policy Example

## Run

From the repository root:

```bash
examples/03_customize_policy/run.sh
```

## Troubleshooting

- `vlastudio: command not found`: from the repository root, `python -m pip install -e .`
- `python: can't open file 'examples/...'`: run `run.sh` from the repository root, not from this directory.
- `Unknown environment: act`: `vlastudio env create` failed; check its output.
- HuggingFace timeout / missing ALOHA hdf5: same as the ACT example; point the task YAML at a local dataset.
- `No module named 'examples/03_customize_policy/my_policy/'`: `unset PYTHONPATH` and rerun so the live checkout is imported, not a cached app snapshot.

## Composition

A custom policy module must export three functions:

- `load_model(args) -> dict`: build or load the model and return at least `{"model": model}`.
- `get_data_processor(args, model_components)`: return a **callable** instance. That callable takes one VLAStudio standard sample and outputs any format that `get_data_collator` can batch.
- `get_data_collator(args, model_components)`: return a batching callable. It receives a list of processor outputs and collates them into a batch.

The processor is the only place that must understand VLAStudio's sample schema. After it, the collator and model only need to agree on the processor's output format.

The object returned by `load_model` must follow Hugging Face's pattern:

- a **config** class that subclasses `transformers.PretrainedConfig`
- a **model** class that subclasses `transformers.PreTrainedModel`
- the model constructor is initialized from that config, for example `MyPolicy(MyPolicyConfig(...))`

The model class must implement two methods:

- `forward(**batch)`: training entry. Hugging Face Trainer calls `model(**collator_output)`, so the keyword arguments of `forward` are the collator batch, including `action` / `is_pad` when they exist. Return a dict that contains `loss`.
- `select_action(batch_obs)`: inference entry. It does **not** receive a raw VLAStudio sample. Evaluation first turns env observations into standard samples, then runs the same `data_processor` and `data_collator` as training, then calls `select_action` with that collated observation batch (typically without `action` / `is_pad`). Return a predicted action chunk.

Attach the processor and collator on the model in `load_model` when loading a checkpoint (`model.data_processor`, `model.data_collator`) so evaluation can reuse that pipeline.

Point the policy YAML `type` at the module that provides those three functions:

```yaml
type: examples/03_customize_policy/my_policy/
```

A directory loads that package's `__init__.py`. A `.py` file can be given directly, with an optional `:Class` suffix.

