# EE559_EPFL_project2


A mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.



## Implmented modules

### Layers
- Linear Layer 
- Conv1d Layer
- BatchNorm Layer
  
### Activation
- Tanh
- Sigmod
- ReLU

### Loss

- MSE
- L2Loss
- L1Loss

### Container Module
- Sequential

## Dependency

cf requirement.txt

## Reproduction

```shell
python3 test.py
```
This will train the model for 400 epochs, leading to
- test F1 score = 0.988
- train F1 score = 0.992
- Normalized MSE Loss = 0.0725
- Total Tranining Time = 53.16 s