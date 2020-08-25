import torch

# TODO: Rewrite me in C++!


def _moveaxis(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)


def chunks(tensor, chunk_size=-1):
    """
    Splits a tensor into evenly-sized chunks of size `chunk_size`. Inserts
    padding as necessary

    >>> eq = lambda a, b: torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-3))
    >>> data = torch.randn(8, 4, 3)
    >>> data_chunked = chunks(data)
    >>> spectral_chunked = dct_n(data_chunked, n=3)
    >>> reconstructed_chunked = idct_n(spectral_chunked, n=3)
    >>> reconstructed = rejoin(reconstructed_chunked, data.shape)

    Okay, so in this example, we take some input data, chunk it, run the chunk
    dimensions through the DCT, run those chunks through the I-DCT and then
    rejoin. Note the `n` parameter on `dct_n` and `idct_n` controls how many
    dimensions to perform the DCT/IDCT on, so we specify the last 3 dimensions.

    >>> data.shape
    torch.Size([8, 4, 3])
    >>> data_chunked.shape
    torch.Size([3, 2, 1, 3, 3, 3])
    >>> spectral_chunked.shape
    torch.Size([3, 2, 1, 3, 3, 3])
    >>> reconstructed_chunked.shape
    torch.Size([3, 2, 1, 3, 3, 3])
    >>> reconstructed.shape
    torch.Size([8, 4, 3])
    >>> eq(data, reconstructed).item()
    True

    :param tensor: data to split into chunks
    :param chunk_size: the size of each chunk, will be the smallest dim if -1
    :return: a tensor split into chunks, batch dimensions first
    """
    chunk_size = min(tensor.shape) if chunk_size == -1 else chunk_size

    # if the specified chunk size is greater than the smallest dimension
    if chunk_size > min(tensor.shape):
        pad_dims = []
        for dim in reversed(tensor.shape):
            pad_dims.append(0)
            pad_dims.append(max(chunk_size - dim, 0))
        padded = torch.nn.functional.pad(tensor, pad_dims)
    else:
        padded = tensor

    arr = padded
    for dimension in range(len(tensor.shape)):
        split_arrays = torch.split(arr, chunk_size, 0)
        joined_arr = torch.nn.utils.rnn.pad_sequence(split_arrays, batch_first=True)

        joined_arr = _moveaxis(joined_arr, 1, len(joined_arr.shape) - 1)

        if dimension < len(tensor.shape) - 1:
            joined_arr = joined_arr.transpose(dimension + 1, 0)
        else:
            joined_arr = _moveaxis(joined_arr, 0, dimension)

        # history.append(joined_arr)
        arr = joined_arr

    return arr


def rejoin(chunked, initial_shape):
    """
    Rejoins chunked tensor, removing the padding as necessary

    >>> eq = lambda a, b: torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-12))
    >>> x = torch.arange(end=4) + 3
    >>> y = torch.arange(end=15) + 2
    >>> mesh = x.view(-1, 1) @ y.view(1, -1)
    >>> mesh = torch.stack([mesh, mesh + 1, mesh + 2], dim=0)

    First we create an array. I don't know why I created it in such a silly way.
    Next, we'll show that chunking/rejoining result in the exact same array,
    despite the fact that some of the chunks are padded!

    >>> mesh.shape
    torch.Size([3, 4, 15])
    >>> chunks(mesh, 3).shape
    torch.Size([1, 2, 5, 3, 3, 3])
    >>> rejoined = rejoin(chunks(mesh, 3), mesh.shape)
    >>> rejoined.shape
    torch.Size([3, 4, 15])
    >>> torch.equal(mesh, rejoined)
    True

    Great! Now we can try specifying a chunk size that is smaller than the
    minimum dimension, and it still works.

    >>> initial = torch.arange(512).view(8, 8, 8)
    >>> chunked = chunks(initial, 9)
    >>> reconstructed = rejoin(chunked, (8, 8, 8))
    >>> torch.equal(initial, reconstructed)
    True

    :param chunked: a chunked tensor created by `chunks`
    :param initial_shape: the initial shape of the tensor before chunking
    :return: tensor in the shape `initial_shape`, dimensions `i` and
        `i + len(initial_shape)` are joined
    """
    indices = []
    padded_shape = []

    for i in range(len(initial_shape)):
        indices.append(i)
        indices.append(i + len(initial_shape))

        padded_shape.append(chunked.shape[i] * chunked.shape[len(initial_shape) + i])

    repermuted = chunked.permute(*indices)
    padded = repermuted.reshape(*padded_shape)

    for i, s in enumerate(initial_shape):
        padded = torch.narrow(padded, i, 0, s)

    return padded


# some examples of the steps in the chunking for 2D/3D

# (a, b)
# (a#, a', b)
# (a#, b, a')
# (b, a#, a')

# (b#, b', a#, a')
# (b#, a#, a', b')
# (a#, b#, a', b')


# (a, b, c)
# (a#, a', b, c)
# (a#, b, c, a')
# (b, a#, c, a')

# (b#, b', a#, c, a')
# (b#, a#, c, a', b')
# (c, a#, b#, a', b')

# (c#, c', a#, b#, a', b')
# (c#, a#, b#, a', b', c')
# (a#, b#, c#, a', b', c')

if __name__ == "__main__":
    import doctest
    from dct import dct_n, idct_n

    doctest.testmod()
