import torch

# Fix for PyTorch >= 2.6 where default weights_only=True breaks OGB dataset loading
_orig_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
