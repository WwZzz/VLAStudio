"""Keep the optional RLDS input pipeline separate from model/tokenizer imports."""


def __getattr__(name):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator
        return get_vla_dataset_and_collator
    raise AttributeError(name)
