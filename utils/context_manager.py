import torch
from contextlib import contextmanager


@contextmanager
def move_to_cuda(device='cuda'):
    """
    Creates all tensors to cuda
    """


    # Save the current device
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else None

    # Move all tensors to the target device
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if device == 'cuda' else 'torch.FloatTensor')

    try:
        yield  # This will execute the code inside the `with` block
    finally:
        # Reset the default tensor type to the original device
        if current_device is not None:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')