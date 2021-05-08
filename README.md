# EE559_EPFL_project2


A mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.



## Activation functions

- Tanh
- Sigmod
- RELU


## Objective functions

- MSE（当函数的输入值距离中心值较远的时候，导致loss 为NAN， 同时也造成使用梯度下降法求解的时候梯度很大，这个与层数无关，
  加一个sigmoid或者tanh 可以解决loss为NAN的问题，但是会造成梯度消失问题，因为sigmoid在 很大输入值的时候梯度几乎=0）
- CrossEntropy


## Layers

- Linear Layer (doesn't have bias right now)
- CNN layer
- softmax layer

## speed
NUMBA (no additional libaray than VM)

## performance compare with Pytorch


## report:

1. Implementation of tanh needs to avoid NaN
```python
    def forward(self, data):
        # TODO This implementation will cause nan, if data is a large value , such as 100

        num = torch.exp(data.double()) - torch.exp(-data.double())
        den = torch.exp(data.double()) + torch.exp(-data.double())

        return num/den
```

