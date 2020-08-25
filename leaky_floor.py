import torch
import numpy as np


def leaky_floor(x, m=0.01):
    floored = torch.floor(x)
    return floored + m * (x - floored)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xvals = np.arange(-2, 3, 0.01)
    yvals = leaky_floor(torch.tensor(xvals)).numpy()
    plt.plot(xvals, yvals)
    plt.show()
