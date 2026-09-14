# Dataset Modules

This directory contains modular dataset implementations for IL-Studio. Each dataset is implemented in its own file for better extensibility and maintainability.

## Structure

```
data_utils/datasets/
├── __init__.py          # Module initialization and exports
├── base.py              # Base EpisodicDataset class
├── aloha_sim.py         # ALOHA simulation datasets
├── aloha_sii.py         # ALOHA SII datasets
├── aloha_sii_v2.py      # ALOHA SII v2 datasets
├── robomimic.py         # RoboMimic benchmark datasets
└── README.md            # This file
```

## Adding New Datasets

To add a new dataset:

1. **Create a new file** (e.g., `my_dataset.py`) in this directory
2. **Import the base class**: `from .base import EpisodicDataset`
3. **Create your dataset class** that inherits from `EpisodicDataset`
4. **Implement required methods**:
   - `get_language_instruction()`: Return task-specific language instruction
   - `load_onestep_from_episode()`: Load single timestep data
   - `load_feat_from_episode()`: Load full episode data
5. **Add to `__init__.py`**: Import and export your new class
6. **Update `data_utils/dataset.py`**: Add import for backward compatibility

## Example

```python
# my_dataset.py
from .base import EpisodicDataset

class MyDataset(EpisodicDataset):
    def get_language_instruction(self):
        return "My custom task instruction"
    
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        # Your custom loading logic
        return {
            'action': action,
            'image': image_dict,
            'state': state,
            'language_instruction': raw_lang,
            'reasoning': reasoning,
        }
    
    def load_feat_from_episode(self, dataset_path, feats=[]):
        # Your custom loading logic
        return data_dict
```

## Benefits

- **Modularity**: Each dataset is self-contained
- **Extensibility**: Easy to add new datasets without modifying existing code
- **Maintainability**: Clear separation of concerns
- **Backward Compatibility**: Existing code continues to work through `data_utils/dataset.py`

# Gallary

## RLDSWrapper


## Robomimic
The robomimic dataset relies on `robomimic`

### Installation
```shell
# use the default env at <path to ILStudio>/.venv/bin/python
uv pip install robomimic
```

### TroubleShooting
If the installation failed due to error like `Compatibility with CMake < 3.5 has been removed from CMake. Update the VERSION argument <min> value.`, please set the configurations by
```shell
export CMAKE_POLICY_VERSION_MINIMUM=X.X # your cmake version
```
And then run the pip install again.

# Libero Tasks
put the white mug on the left plate and put the yellow and white mug on the right plate      0
put the white mug on the plate and put the chocolate pudding to the right of the plate       1
put the yellow and white mug in the microwave and close it                                   2
turn on the stove and put the moka pot on it                                                 3
put both the alphabet soup and the cream cheese box in the basket                            4
put both the alphabet soup and the tomato sauce in the basket                                5
put both moka pots on the stove                                                              6
put both the cream cheese box and the butter in the basket                                   7
put the black bowl in the bottom drawer of the cabinet and close it                          8
pick up the book and place it in the back compartment of the caddy                           9
put the bowl on the plate                                                                   10
put the wine bottle on the rack                                                             11
open the top drawer and put the bowl inside                                                 12
put the cream cheese in the bowl                                                            13
put the wine bottle on top of the cabinet                                                   14
push the plate to the front of the stove                                                    15
turn on the stove                                                                           16
put the bowl on the stove                                                                   17
put the bowl on top of the cabinet                                                          18
open the middle drawer of the cabinet                                                       19
pick up the orange juice and place it in the basket                                         20
pick up the ketchup and place it in the basket                                              21
pick up the cream cheese and place it in the basket                                         22
pick up the bbq sauce and place it in the basket                                            23
pick up the alphabet soup and place it in the basket                                        24
pick up the milk and place it in the basket                                                 25
pick up the salad dressing and place it in the basket                                       26
pick up the butter and place it in the basket                                               27
pick up the tomato sauce and place it in the basket                                         28
pick up the chocolate pudding and place it in the basket                                    29
pick up the black bowl next to the cookie box and place it on the plate                     30
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate    31
pick up the black bowl on the ramekin and place it on the plate                             32
pick up the black bowl on the stove and place it on the plate                               33
pick up the black bowl between the plate and the ramekin and place it on the plate          34
pick up the black bowl on the cookie box and place it on the plate                          35
pick up the black bowl next to the plate and place it on the plate                          36
pick up the black bowl next to the ramekin and place it on the plate                        37
pick up the black bowl from table center and place it on the plate                          38
pick up the black bowl on the wooden cabinet and place it on the plate                      39