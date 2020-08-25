import numba as nb
import numpy as np

@nb.njit("uint64(uint8, uint8, uint8)")
def separate_n_nb(packed, n, chunk_bits):
    """
    A relatively inefficient generalization of the "separate bits"
    step of Morton encoding. Assuming that each of the `n` coordinates
    has `chunk_bits` bits, we can "space out" each bit of each coordinate
    `n` spaces at a time.

    >>> for i in range(8):
    ...     print(i,
    ...           format(separate_n_nb(i, 3, 3), '#012b'),
    ...           format(separate_n_nb(i, 3, 3) << 1, '#012b'),
    ...           format(separate_n_nb(i, 3, 3) << 2, '#012b'))
    0 0b0000000000 0b0000000000 0b0000000000
    1 0b0000000001 0b0000000010 0b0000000100
    2 0b0000001000 0b0000010000 0b0000100000
    3 0b0000001001 0b0000010010 0b0000100100
    4 0b0001000000 0b0010000000 0b0100000000
    5 0b0001000001 0b0010000010 0b0100000100
    6 0b0001001000 0b0010010000 0b0100100000
    7 0b0001001001 0b0010010010 0b0100100100

    :param packed: packed tensor
    :param n: number of components that we will eventually want to Morton code
    :param chunk_bits: the number of bits that represent each coordinate
    :return: spaced-out bit representation, ready to be interleaved
    """
    a = nb.uint64(packed)

    a = a & nb.uint64(0x00000000000000FF)

    x = 0
    for i in range(chunk_bits):
        bit_to_set = nb.uint64(1) << nb.uint64(i * n)
        x |= (a << nb.uint64((n - 1) * i)) & bit_to_set

    return x


@nb.njit("uint8(uint64, uint8, uint8)")
def pack_n_nb(spaced, n, chunk_bits):
    a = 0
    for i in range(chunk_bits):
        bit_idx = nb.uint64(i * n)
        bit_to_set = nb.uint64(1) << bit_idx
        a |= (spaced & bit_to_set) >> (bit_idx - i)

    return a


@nb.njit("uint64(uint8[:], uint8)")
def encode_single_coord(coord, chunk_bits):
    """
    Encodes a coordinate in ℝⁿ in ℝ¹ using Morton ordering, assuming that
    the size of each dimension is 0..2^{chunk_bits}

    >>> morton_offsets = set()
    >>> for i in range(16):
    ...     for j in range(16):
    ...         morton_offsets.add(encode_single_coord(
    ...                             np.array([i, j], dtype=np.uint8),
    ...                             4))
    >>> morton_offsets == {i for i in range(256)}
    True

    Here we demonstrate that there is mapping from coordinates in a 16x16 square
    to the numbers 0..255

    :param coord: coordinate to encode, numba array of type uint8, size <= 8
    :param chunk_bits: coordinate dimensions
    :return: Morton-coded offset of type uint64
    """
    assert coord.shape[0] <= 8
    x = nb.uint64(0)
    for i in range(coord.shape[0]):
        x += separate_n_nb(coord[i], coord.shape[0], chunk_bits) << i

    return x


@nb.njit("uint8[:](uint64, uint8, uint8)")
def decode_single_coord(offset, n, chunk_bits):
    """
    The reverse of the Morton encode function above

    >>> verify_decode = set()
    >>> for i in range(16):
    ...     for j in range(16):
    ...         coord = np.array([i, j], dtype=np.uint8)
    ...         encoded = encode_single_coord(coord, 4)
    ...         decoded = decode_single_coord(encoded, 2, 4)
    ...         verify_decode.add(np.array_equal(coord, decoded))
    >>> all(v for v in verify_decode)
    True

    :param offset: morton encoded offset
    :param n: dimensionality of coordinates
    :param chunk_bits: size of the coordinate dimensions
    """
    coord = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        coord[i] = pack_n_nb(offset >> i, n, chunk_bits)

    return coord


@nb.njit
def log2i(s):
    a = 0
    tmp = s
    while tmp > 0:
        tmp = tmp >> 1
        a += 1

    return a


@nb.njit
def morton_encode_nb(coords):
    """
    >>> x, y = np.arange(8), np.arange(8)
    >>> xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    >>> inp = np.sqrt(xv ** 2 + yv ** 2).reshape(1, 8, 8)

    For the sake of clarity, let's inspect these values

    >>> with np.printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}):
    ...     print(inp)
    [[[0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000]
      [1.000 1.414 2.236 3.162 4.123 5.099 6.083 7.071]
      [2.000 2.236 2.828 3.606 4.472 5.385 6.325 7.280]
      [3.000 3.162 3.606 4.243 5.000 5.831 6.708 7.616]
      [4.000 4.123 4.472 5.000 5.657 6.403 7.211 8.062]
      [5.000 5.099 5.385 5.831 6.403 7.071 7.810 8.602]
      [6.000 6.083 6.325 6.708 7.211 7.810 8.485 9.220]
      [7.000 7.071 7.280 7.616 8.062 8.602 9.220 9.899]]]

    We rearrange them according to the Morton encoding, and then reshape
    the resulting array into the same dimensions as the chunks. You can see
    that (for the most part) groups similar frequency ranges close to each
    other (TODO: perhaps this isn't the best visualization...)

    >>> with np.printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}):
    ...     print(morton_encode_nb(inp)[0].reshape(8, 8))
    [[0.000 1.000 1.000 1.414 2.000 3.000 2.236 3.162]
     [2.000 2.236 3.000 3.162 2.828 3.606 3.606 4.243]
     [4.000 5.000 4.123 5.099 6.000 7.000 6.083 7.071]
     [4.472 5.385 5.000 5.831 6.325 7.280 6.708 7.616]
     [4.000 4.123 5.000 5.099 4.472 5.000 5.385 5.831]
     [6.000 6.083 7.000 7.071 6.325 6.708 7.280 7.616]
     [5.657 6.403 6.403 7.071 7.211 8.062 7.810 8.602]
     [7.211 7.810 8.062 8.602 8.485 9.220 9.220 9.899]]

    :param coords: coords is [BS, chunk1, chunk2, chunk3...]. All chunks
        must have the same size!
    :return: rearranged array of size [BS, product of chunk sizes]
    """
    assert len(coords.shape) <= 9
    # assert all(coord == coords[1] for coord in coords[1:])
    n = len(coords.shape) - 1

    bs = coords.shape[0]
    chunk_size = nb.uint8(coords.shape[n])
    total_chunk_size = nb.int64(chunk_size ** n)

    chunk_bits = log2i(chunk_size - 1)

    ind = np.zeros(n, dtype=nb.uint8)

    output = np.zeros((bs, total_chunk_size), dtype=coords.dtype)

    for index, x in np.ndenumerate(coords):
        for i in range(n):
            ind[i] = np.uint8(index[1 + i])
        morton_offset = encode_single_coord(ind, chunk_bits)
        output[index[0], morton_offset] = x

    return output


@nb.njit
def morton_decode_nb(offsets, output):
    """
    >>> x, y = np.arange(64), np.arange(64)
    >>> xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    >>> inp = np.sqrt(xv ** 2 + yv ** 2).reshape(1, 64, 64)
    >>> recon = np.zeros_like(inp)

    >>> morton_decode_nb(morton_encode_nb(inp), recon)
    >>> (inp - recon).max() < 1e-5
    True

    This function is basically the inverse of `morton_encode_nb`, converting
    from 1D offsets to ND coordinates. It stores the results in `output

    :param offsets: [BS, product of chunk sizes] encoded offsets
    :param output: output is [BS,  chunk1, chunk2, chunk3...]. All chunks must
        have the same size!
    """
    bs = offsets.shape[0]
    assert bs == output.shape[0]

    n = len(output.shape) - 1
    assert n <= 8

    chunk_size = output.shape[1]
    ind = np.zeros(n, dtype=nb.uint8)

    chunk_bits = log2i(chunk_size - 1)

    for index, _ in np.ndenumerate(output):
        for i in range(n):
            ind[i] = np.uint8(index[1 + i])
        morton_offset = encode_single_coord(ind, chunk_bits)

        output[index] = offsets[index[0], morton_offset]

if __name__ == "__main__":
    import doctest

    doctest.testmod()
