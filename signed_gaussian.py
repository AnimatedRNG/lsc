import torch
import numpy as np


def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * torch.exp(
        (-1 / 2) * torch.square((x - mu) / (sigma))
    )


def signed_gaussian_pdf(x, mu, sigma):
    return (
        (1 / (sigma * np.sqrt(2 * np.pi)))
        * torch.exp((-1 / 2) * torch.square((x - mu) / (sigma)))
        * torch.sign(x - mu)
    )


def reverse_gaussian_pdf(y, mu, sigma):
    """
    Not the \"inverse gaussian\", but inverses the gaussian function

    >>> eq = lambda a, b: torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-4))
    >>> x = abs(torch.rand((100,)) + 5.0)
    >>> gaussian_output = gaussian_pdf(x, 5.0, 1.0)
    >>> rev_x = reverse_gaussian_pdf(gaussian_output, 5.0, 1.0)
    >>> eq(x, rev_x).item()
    True

    :param y output of the gaussian
    :param mu mean
    :param sigma std dev
    """
    return torch.sqrt(torch.log(y * (sigma * np.sqrt(2 * np.pi))) * (-2)) * sigma + mu


def reverse_signed_gaussian_pdf(y, mu, sigma):
    """
    Not the \"inverse gaussian\", but inverses the signed gaussian function

    >>> eq = lambda a, b: torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-1))
    >>> x = abs(torch.rand((100,)) * 5.0 - 2.5)
    >>> gaussian_output = signed_gaussian_pdf(x, 2.5, 10.0)
    >>> rev_x = reverse_signed_gaussian_pdf(gaussian_output, 2.5, 10.0)
    >>> eq(x, rev_x).item()
    True

    :param y output of the gaussian
    :param mu mean
    :param sigma std dev
    """
    return (
        torch.sqrt(torch.log(abs(y) * (sigma * np.sqrt(2 * np.pi))) * (-2))
        * sigma
        * torch.sign(y)
        + mu
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
