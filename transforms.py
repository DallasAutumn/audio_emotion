from torchvision.transforms import Compose
import torch
import torch.nn as nn


class ToTensor:
    def __call__(self, mfcc):
        return torch.from_numpy(mfcc)


class Normalize:
    def __call__(self, tensor):
        tensor_minusmin = tensor - tensor.min()
        return tensor_minusmin / tensor_minusmin.abs().max()


class PaddingSame2d:
    """
    Padding to the same given sequence length.
    """

    def __init__(self, seq_len=224, value=0):
        assert isinstance(seq_len, int)

        self.seq_len = seq_len
        self.value = value

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor)

        pad = nn.ConstantPad2d(
            padding=(0, self.seq_len-tensor.shape[1], 0, 0),
            value=self.value
        )
        return pad(tensor)
