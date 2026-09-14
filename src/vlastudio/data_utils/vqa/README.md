# VQA module

This module implements Vision Query Answer datasets

# Format 

Each dataset's element should be a dict like
```python
{
    'image': torch.Tensor((n,c,h,w), dtype=torch.uint8),
    'raw_lang': str, # can be prompt of instruction, or the type of VQA, or empty string if not necessary
    'reasoning': dict(
        'conversation': [
            dict(from=..., value=...),
            dict(from=..., value=...),
            ...
        ],
        'meta': dict, # can contains 
        'is_vqa': True
    ),
    
}
```

Then, a wrapper will wrap each item by another dummy dict to align its format with the standard ilstudio sample format

```python
additional part = {
    'state': state_data, # torch.Tensor(state_dim, )
    'action': action_data, # torch.Tensor(chunk_size, action_dim)
    'is_pad': is_pad, # torch.Tensor(chunk_size, ), dtype=bool
    'timestamp': 0, 
    'episode_id': 0,
    '__index__': int,
}
```

This wrapper initializes a common dummy `state`, `action`, and `is_pad` (e.g., True for all the values) in its __init__ function and then equip each item with these variables.
