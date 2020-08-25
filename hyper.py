import torch

class WrapModule(torch.nn.Module):
    def __init__(self, m):
        super(WrapModule, self).__init__()
        self.m = m

    def forward(self, *args, **kwargs):
        return self.m(*args, **kwargs)


def wrap_modules(m, wc, depth=0):
    """
    Wraps a PyTorch module, mutating it in-place

    >>> import torch
    >>> import torchvision.models as models
    >>> eq = lambda a, b: torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-12))
    >>> resnet18 = models.resnet18(pretrained=True)
    >>> wrapped_resnet18 = models.resnet18(pretrained=True)
    >>> wrap_modules(wrapped_resnet18, WrapModule)

    >>> random_image = torch.randn(1, 3, 128, 128)
    >>> eq(resnet18(random_image), wrapped_resnet18(random_image)).item()
    True

    :param m the module to wrap
    :param wc the wrapper module
    """
    for attr_name in dir(m):
        attr = getattr(m, attr_name)

        if isinstance(attr, torch.nn.Module):
            setattr(m, attr_name, wrap_modules(attr, wc, depth + 1))
        if isinstance(attr, torch.nn.ModuleList):
            setattr(
                m,
                attr_name,
                torch.nn.ModuleList(
                    (
                        *list(
                            wrap_modules(child, wc, depth + 2)
                            for child in attr.children()
                        )
                    )
                ),
            )
        if isinstance(attr, torch.nn.Sequential):
            setattr(
                m,
                attr_name,
                torch.nn.Sequential(
                    (
                        *list(
                            wrap_modules(child, wc, depth + 2)
                            for child in attr.children()
                        )
                    )
                ),
            )

    if isinstance(m, torch.nn.Module) and depth > 0:
        return wc(m)


class HyperNetworkExample(torch.nn.Module):
    def __init__(self, e):
        super(HyperNetworkExample, self).__init__()
        self.e_shape = e.shape
        self.d = torch.nn.Parameter(torch.ones([1], dtype=torch.float32))

    def forward(self):
        return torch.pow(self.d.expand(self.e_shape), 2.0)


class HyperNetwork(torch.nn.Module):
    """
    Given a model `m` parameters :math:`\theta` which can be described as
    :math:`m(\theta, x)`, this module represents :math:`m(f(), x)`

    >>> eq = lambda a, b: torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-1))
    >>> model = torch.nn.Linear(5, 5)
    >>> hyper = HyperNetwork(model, HyperNetworkExample, ('weight', 'bias'))
    >>> hyper.train()
    >>> optim = torch.optim.Adam(hyper.parameters(), lr=1e-3)
    >>> for i in range(5000):
    ...     out = hyper(torch.randn([5]))
    ...     loss = ((out - 16.0) ** 2).mean()
    ...     optim.zero_grad()
    ...     loss.backward()
    ...     optim.step()
    >>> d = {param_name: param for param_name, param in hyper.named_parameters()}['d']
    >>> eq(d, 4.0).item()
    True

    :param m the primary network
    :param f_cls the class of the hypernetwork (i.e the network that generates
        `m`'s weights)
    """

    def __init__(self, m, f_cls, attrs):
        super(HyperNetwork, self).__init__()

        self.m = m
        self.fs = []
        self.attrs = [
            attr
            for attr in attrs
            if hasattr(self.m, attr) and getattr(self.m, attr) is not None
        ]

        for attr in self.attrs:
            f = f_cls(getattr(self.m, attr))

            for name, parameter in f.named_parameters():
                self.register_parameter(name, parameter)

            self.fs.append(f)

            if attr in self.m._parameters:
                del self.m._parameters[attr]

    def forward(self, *args):
        for i, attr in enumerate(self.attrs):
            f = self.fs[i]
            setattr(self.m, attr, f())

        return self.m(*args)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
