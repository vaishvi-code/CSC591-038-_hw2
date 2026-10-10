import torch
import builtins

# Fix for PyTorch >= 2.6 where default weights_only=True breaks OGB dataset loading
_orig_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Auto-accept OGB interactive update prompts (e.g. 'Will you update the dataset now? (y/N)')
_orig_input = builtins.input

def _auto_input(prompt=''):
    print(f"{prompt} -> Auto-selected: y")
    return 'y'

builtins.input = _auto_input

