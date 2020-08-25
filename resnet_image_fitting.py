import torch
import numpy as np
import torchvision.models as models
from zstd import ZSTD_compress

from spectral import *

resnet18 = models.resnet18(pretrained=True).cuda()
compressed_resnet18 = models.resnet18(pretrained=True)
#resnet18 = models.resnet152(pretrained=True).cuda()
#compressed_resnet18 = models.resnet152(pretrained=True)
compressed_resnet18.cuda()

q_net = spectral(compressed_resnet18)

optim = torch.optim.Adam(list(compressed_resnet18.parameters()) + list(q_net.parameters()), lr=1e-3)

image = torch.randn((1, 3, 224, 224), dtype=torch.float32).cuda()

ref = resnet18(image).detach()

for i in range(10000):
    if i % 20 == 0:
        retarget(compressed_resnet18)

    print('iteration', i)

    optim.zero_grad()

    y = compressed_resnet18(image)

    reconstruction_loss = ((ref - y) ** 2).mean()
    q_loss = quantization_loss(compressed_resnet18)
    loss = reconstruction_loss + q_loss

    loss.backward(retain_graph=False)

    optim.step()

    if i % 10:
        with torch.no_grad():
            model_size_mb = pre_entropy_size(compressed_resnet18).item() / (8 * 2 ** 20)
            pack_size = len(ZSTD_compress(compress_weights(compressed_resnet18)['packs'].numpy().tobytes()))
            pack_size_mb = pack_size / (2 ** 20)

            print('reconstruction', reconstruction_loss)
            print('q_loss', q_loss)
            print('size', model_size_mb, 'MB')
            print('pack size', pack_size_mb, 'MB')
