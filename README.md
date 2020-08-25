# lsc: Learned Spectral Compression

To convert a model to the spectral representation, just run

```python
from lsc import spectral, quantization_loss

q_net = spectral(my_model)
```

`spectral` will convert `my_model` in-place. `q_net` is the quantization network. This network learns how to compress the model during the optimization process. The user does not need to do anything with it other than pass `q_net.parameters()` into their optimizer during the fine-tuning or training step. Admittedly the quantization network does add to the memory/space footprint of the model overall, but only by ~10 KB.

```python
optim = torch.optim.Adam(
    list(my_model.parameters()) + list(q_net.parameters()), 
    lr=1e-3
)
```

Next, during the optimization loop, the user needs to incorporate the _quantization loss_ into their training process. The quantization loss represents the average number of bits per spectral weight in the model. By default, this value starts at 10 bits of precision for all weights (which is already a ~70% improvement over most float32 models). Depending on the use-case, the user might want to modify the output of this loss function (perhaps to limit compression beyond a certain point).

```python
my_usual_model_loss = ...
q_loss = quantization_loss(my_model)
loss = my_usual_model_loss + q_loss

loss.backward()
optim.step()
```

After training, the user can run the final entropy coding stage and extract a `state_dict` using the `compress_weights` function. Ideally we wouldn't need a custom `compress_weights` function (and the compression would just happen during `state_dict()`) but I haven't figured that part out yet.

```python
my_state_dict = compress_weights(my_model)
```

*TODO: Add some more experiments...*
