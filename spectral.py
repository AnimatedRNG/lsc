import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
import numba as nb
from collections import OrderedDict

from hyper import HyperNetwork, wrap_modules
from quantization import QuantizationMLP
from leaky_floor import leaky_floor
from chunks import chunks, rejoin
from signed_gaussian import signed_gaussian_pdf, reverse_signed_gaussian_pdf
from morton import morton_encode_nb, morton_decode_nb
from dct import dct_n, idct_n


def spectral_to_amplitude(
    spectral_representation, tensor_mu, tensor_std, quantization_tensor, weight_shape
):
    qt = pow(2.0, quantization_tensor)
    compressed_spectral_representation = leaky_floor(spectral_representation * qt) / qt

    amplitude_representation = rejoin(
        idct_n(compressed_spectral_representation, n=len(weight_shape)), weight_shape
    )

    return signed_gaussian_pdf(amplitude_representation, tensor_mu, tensor_std)


@nb.njit
def into_bitset(data, quantization_tensor, max_bits_per_element):
    output = np.zeros(data.shape[0] * max_bits_per_element, dtype=np.uint8)

    idx = 0
    for a, nb in np.ndenumerate(quantization_tensor):
        for i in range(nb):
            r_val = 1 if (data[a] & (1 << i)) > 0 else 0
            output[idx] = r_val
            idx += 1

    output = output[:idx]

    return output


@nb.njit
def from_bitset(bitset, quantization_tensor, max_bits_per_element, output):
    """
    >>> data = np.arange(62, dtype=np.int8)
    >>> recon = np.zeros_like(data, dtype=np.int8)
    >>> qt = np.ones(62, dtype=np.int8) * 8
    >>> bs = into_bitset(data, qt, 8)
    >>> from_bitset(bs, qt, 8, recon)
    >>> np.array_equal(data, recon)
    True
    """
    ind = 0
    for a, nb in np.ndenumerate(quantization_tensor):
        for i in range(max_bits_per_element):
            output[a] |= bitset[ind] << i
            ind += 1


class SpectralCompressionModule(torch.nn.Module):
    def __init__(self, e):
        super(SpectralCompressionModule, self).__init__()

        with torch.no_grad():
            self.weight_shape = e.shape
            self.chunk_size = self.optimal_chunk_size(self.weight_shape)

            self.register_buffer("tensor_mu", e.mean())
            self.register_buffer("tensor_std", e.std())

            gaussian_distributed = reverse_signed_gaussian_pdf(
                e, self.tensor_mu, self.tensor_std
            )

            self.spectral_representation = torch.nn.Parameter(
                dct_n(
                    chunks(gaussian_distributed, self.chunk_size),
                    n=len(self.weight_shape),
                )
            )

        self.spectral_representation.requires_grad_(True)

    def set_quantization_network(self, quantization_network, module_id, zero_cliff=0.8):
        self.quantization_network = quantization_network

        module_id_ten = torch.tensor(
            [module_id], dtype=torch.float32, device=self.spectral_representation.device
        )
        self.register_buffer("module_id", module_id_ten)
        self.zero_cliff = zero_cliff

    def compute_quantization_tensor(self):
        # [a, b, c, d]
        # [1, 1, 1, 1, d, c, b, a]
        # [8, 7, 6, 5, 4, 3, 2, 1]

        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        num_dims = len(self.spectral_representation.shape)
        compress_shape = self.spectral_representation.shape[num_dims // 2 :]
        # compress_shape = self.spectral_representation.shape

        empty_dimensions = 8 - len(compress_shape)
        quantization_tensor = self.quantization_network(
            self.module_id, (1,) * empty_dimensions + tuple(reversed(compress_shape))
        )
        """quantization_dims = \
            (1,) * empty_dimensions + tuple(reversed(compress_shape))
        quantization_tensor = checkpoint(
            self.quantization_network,
            self.module_id,
            torch.tensor(quantization_dims, dtype=torch.int64), # needed for "save for backwards"
            dummy_tensor
        )"""
        quantization_tensor = quantization_tensor.permute(
            tuple(reversed(range(8)))
        ).view(compress_shape)

        quantization_tensor = quantization_tensor.expand(
            self.spectral_representation.shape
        )

        return torch.clamp(quantization_tensor, min=self.zero_cliff)

    def forward(self, *args, **kwargs):
        assert not (self.quantization_network is None)
        assert not (self.module_id is None)

        quantization_tensor = self.compute_quantization_tensor()
        compressible_tensor = spectral_to_amplitude(
            self.spectral_representation,
            self.tensor_mu,
            self.tensor_std,
            quantization_tensor,
            self.weight_shape,
        )

        self.cached_quantization_loss = self.quantization_loss(quantization_tensor)

        self.cached_quantization_size = self.size(quantization_tensor)

        return compressible_tensor

    def pack(self):
        with torch.no_grad():
            quantization_tensor = self.compute_quantization_tensor()

            # compute the highest bit-depth of any entry of the quantization tensor
            max_bits = torch.ceil(quantization_tensor.max()).item()

            # get the size of the integer we need to use
            int_type = int(2 ** np.ceil(np.log2(max_bits)))

            assert int_type <= 64

            buf_type = None
            if int_type == 64:
                buf_type = torch.int64
            elif int_type == 32:
                buf_type = torch.int32
            elif int_type == 16:
                buf_type = torch.int16
            else:
                buf_type = torch.int8

            # requantize
            qt = pow(2.0, quantization_tensor)
            compressed_spectral_representation = torch.floor(
                self.spectral_representation * qt
            )
            integer_spectral_representation = compressed_spectral_representation.type(
                buf_type
            )

            batch_dims = len(integer_spectral_representation.shape) // 2
            batched_isr = (
                integer_spectral_representation.view(
                    -1, *integer_spectral_representation.shape[batch_dims:]
                )
                .cpu()
                .numpy()
            )

            # morton coding
            morton_encoded = (
                morton_encode_nb(batched_isr)
                if len(batched_isr.shape) > 2
                else batched_isr
            )
            morton_encoded = morton_encoded.flatten()

            quantization_tensor = (
                quantization_tensor.type(torch.uint8).cpu().numpy().flatten()
            )

            bitset_repr = into_bitset(morton_encoded, quantization_tensor, int_type)
            packed = np.packbits(bitset_repr)

            return packed

    def optimal_chunk_size(self, weight_dims):
        smallest_dimension = min(weight_dims)
        return min(int(2 ** int(np.ceil(np.log2(smallest_dimension)))), 128)

    def retarget(self):
        with torch.no_grad():
            if hasattr(self, "cached_quantization_tensor"):
                quantization_tensor = self.cached_quantization_tensor
            else:
                quantization_tensor = self.compute_quantization_tensor()

            compressible_tensor = spectral_to_amplitude(
                self.spectral_representation,
                self.tensor_mu,
                self.tensor_std,
                quantization_tensor,
                self.weight_shape,
            )

            new_tensor_mu, new_tensor_std = (
                compressible_tensor.mean(),
                compressible_tensor.std(),
            )

            gaussian_distributed = reverse_signed_gaussian_pdf(
                compressible_tensor, new_tensor_mu, new_tensor_std
            )

            self.spectral_representation.copy_(
                dct_n(
                    chunks(gaussian_distributed, self.chunk_size),
                    n=len(self.weight_shape),
                )
            )

            self.tensor_mu, self.tensor_std = new_tensor_mu, new_tensor_std

            # print('old tensor mu', self.tensor_mu, 'old tensor std', self.tensor_std,
            #      'new tensor mu', new_tensor_mu, 'new_tensor_std', new_tensor_std)

    def quantization_loss(self, quantization_tensor):
        return torch.abs(quantization_tensor).mean()

    def size(self, quantization_tensor):
        return torch.floor(quantization_tensor).sum()


def init_quantization_network(parent):
    sps = [
        sp
        for child in parent.modules()
        if isinstance(child, HyperNetwork)
        for sp in child.fs
    ]

    network = QuantizationMLP(len(sps)).to(sps[0].spectral_representation.device)

    for q_id, sp in enumerate(sps):
        sp.set_quantization_network(network, q_id)

    return network


def quantization_loss(parent):
    losses = [
        f.cached_quantization_loss
        for child in parent.modules()
        if isinstance(child, HyperNetwork)
        for f in child.fs
    ]
    return sum(losses) / len(losses)


def retarget(parent):
    for child in parent.modules():
        if isinstance(child, HyperNetwork):
            for f in child.fs:
                f.retarget()


def pre_entropy_size(parent):
    return sum(
        f.cached_quantization_size
        for child in parent.modules()
        if isinstance(child, HyperNetwork)
        for f in child.fs
    )


def compress_weights(parent):
    with torch.no_grad():
        sd = parent.state_dict()
        compressed_sd = {}

        for name, val in sd.items():
            if (
                name.find("spectral_representation") == -1
                and name.find("quantization_network") == -1
                and name.find("cached_") == -1
            ):
                compressed_sd[name] = val

        modules = {}
        for child in parent.modules():
            if isinstance(child, HyperNetwork):
                for f in child.fs:
                    module_id = int(f.module_id[0].item())
                    # compressed_sd['{}.pack'.format(module_id)] = f.pack()
                    modules[module_id] = f.pack()

        modules_list = [None for _ in range(len(modules))]
        modules_shapes = []
        for module_id, pack in modules.items():
            modules_list[module_id] = torch.from_numpy(pack)
            modules_shapes.append(pack.shape[0])

        compressed_sd["packs"] = torch.cat(modules_list, dim=0)
        compressed_sd["packs_shapes"] = torch.tensor(modules_shapes, dtype=torch.int64)

    return compressed_sd


def spectral(network, compress_params=("weight", "bias")):
    hyp = lambda m: HyperNetwork(m, SpectralCompressionModule, compress_params)
    wrap_modules(network, hyp)
    network.train()
    return init_quantization_network(network)
