import torch
from torch.utils.checkpoint import checkpoint

from quantization import QuantizationMLP

gt = [
    torch.randn(1024),
    torch.randn(256),
    torch.randn(512),
    torch.randn(256, 256, 4, 4),
    torch.randn(128, 128, 4, 4),
]

network = QuantizationMLP(len(gt))
network = network

optim = torch.optim.Adam(network.parameters(), lr=1e-3)

dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

for i in range(2000):
    for j in range(len(gt)):
        ref = gt[j]

        # s = network(torch.tensor([j]), ref.shape)
        s = checkpoint(
            network,
            torch.tensor([j]),
            torch.tensor(ref.shape, dtype=torch.int64),
            dummy_tensor,
        )

        loss = ((ref - s) ** 2).mean()
        print(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()
