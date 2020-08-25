import torch
import numpy as np


def positional_encoding(x, L=2):
    assert len(x.shape) == 2

    ls = torch.floor(
        torch.arange(0, L, step=0.5, dtype=x.dtype, device=x.device)
    ).repeat(x.shape + (1,))
    gamma = torch.zeros_like(ls, dtype=x.dtype, device=x.device)

    gamma[:, :, ::2] = torch.sin(torch.pow(2, ls[:, :, ::2]) * np.pi * x.unsqueeze(-1))
    gamma[:, :, 1::2] = torch.cos(
        torch.pow(2, ls[:, :, 1::2]) * np.pi * x.unsqueeze(-1)
    )

    return gamma


class QuantizationMLP(torch.nn.Module):
    def __init__(
        self, num_modules, quantization_bias=10.0, input_features=64, num_features=32
    ):
        super(QuantizationMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_features, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, 1),
            torch.nn.ReLU(),
        )
        self.quantization_bias = quantization_bias
        self.num_modules = num_modules
        self.num_features = num_features
        self.input_features = input_features

    def forward(self, module_id, weights_shape, *args):
        assert len(module_id.shape) == 1
        assert module_id.shape[0] == 1

        if isinstance(weights_shape, torch.Tensor):
            weights_shape = tuple(weights_shape.tolist())

        position_gen = (
            torch.arange(d, dtype=module_id.dtype, device=module_id.device)
            for d in weights_shape
        )

        # replace with linspace?
        pos = torch.stack(torch.meshgrid((module_id,) + tuple(position_gen)), dim=-1)

        mod_weights_shape = (self.num_modules,) + weights_shape

        weight_normalization = torch.tensor(
            mod_weights_shape, dtype=module_id.dtype, device=pos.device
        )

        pos_norm = (pos + 0.5) / weight_normalization

        # assert (pos_norm < 1.0).all()

        pos_linear = pos_norm.view(-1, len(mod_weights_shape))

        encoded = positional_encoding(pos_linear)
        encoded = encoded.view(encoded.shape[0], -1)
        pad_amount = self.input_features - encoded.shape[-1]
        assert pad_amount >= 0
        encoded = torch.nn.functional.pad(encoded, (0, pad_amount))
        """encoded = pos_linear.view(pos_linear.shape[0], -1)
        pad_amount = self.input_features - encoded.shape[-1]
        assert pad_amount >= 0
        encoded = torch.nn.functional.pad(encoded,
                                          (0, pad_amount))"""

        quantization = self.layers(encoded)

        return self.quantization_bias - quantization.view(weights_shape)
