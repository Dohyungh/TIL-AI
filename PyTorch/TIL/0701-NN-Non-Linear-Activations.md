# Non-linear Activations (weighted sum, nonlinearity)

1. Tensor  
2. Dataset,DataLoader  
   1. `torch.utils.data`
3. Transform
   1. `torchvision.transforms.v2`
      1. `v2.CutMix`
      2. `v2.MixUp`
   2. `torchvision.transforms`
4. nn-model  
   1. `torch.nn`
        1. Convolution Layers
            1. Convolution
               1. `nn.Conv#d`
               2. `nn.ConvTranspose#d`
        2. Pooling Layers
        3. **Non-Linear Activations**
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---

## nn.Non-Linear Activations (weighted sum, nonlinearity)
- `nn.ELU` : Applies the Exponential Linear Unit (ELU) function, element-wise.
- `nn.Hardshrink` : Applies the Hard Shrinkage (Hardshrink) function element-wise.
- `nn.Hardsigmoid` : Applies the Hardsigmoid function element-wise.
- `nn.Hardtanh` : Applies the HardTanh function element-wise.
- `nn.Hardswish` : Applies the Hardswish function, element-wise.
- `nn.LeakyReLU` : Applies the LeakyReLU function element-wise.
- `nn.LogSigmoid` : Applies the Logsigmoid function element-wise.
- `nn.MultiheadAttention` : Allows the model to jointly attend to information from different representation subspaces.
- `nn.PReLU` : Applies the element-wise PReLU function.
- `nn.ReLU` : Applies the rectified linear unit function element-wise.
- `nn.ReLU6` : Applies the ReLU6 function element-wise.
- `nn.RReLU` : Applies the randomized leaky rectified linear unit function, element-wise.
- `nn.SELU` : Applies the SELU function element-wise.
- `nn.CELU` : Applies the CELU function element-wise.
- `nn.GELU` : Applies the Gaussian Error Linear Units function.
- `nn.Sigmoid` : Applies the Sigmoid function element-wise.
- `nn.SiLU` : Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
- `nn.Mish` : Applies the Mish function, element-wise.
- `nn.Softplus` : Applies the Softplus function element-wise.
- `nn.Softshrink` : Applies the soft shrinkage function element-wise.
- `nn.Softsign` : Applies the element-wise Softsign function.
- `nn.Tanh` : Applies the Hyperbolic Tangent (Tanh) function element-wise.
- `nn.Tanhshrink` : Applies the element-wise Tanhshrink function.
- `nn.Threshold` : Thresholds each element of the input Tensor.
- `nn.GLU` : Applies the gated linear unit function.

## Non-linear Activations (other)

- `nn.Softmin` : Applies the Softmin function to an n-dimensional input Tensor.
- `nn.Softmax` : Applies the Softmax function to an n-dimensional input Tensor.
- `nn.Softmax2d` : Applies SoftMax over features to each spatial location.
- `nn.LogSoftmax` : Applies the log(Softmax(𝑥)) function to an n-dimensional input Tensor.
- `nn.AdaptiveLogSoftmaxWithLoss` : Efficient softmax approximation.


## Vanishing Gradient

[기울기 소실(Vanishing Gradient)의 의미와 해결방법](https://heytech.tistory.com/388)

## Exponential Linear Unit (ELU) function, element-wise.

[FAST AND ACCURATE DEEP NETWORK LEARNING BY
EXPONENTIAL LINEAR UNITS (ELUS)](./assets/0701ELU.pdf)

### definition

$$
ELU(x) = \begin{cases}
   x &\text{if } x\gt0 \\
   \alpha * (exp(x)-1) &\text{if } x \leq 0
\end{cases}
$$



### ELU, LReLU, ReLU, SReLU 비교
<p align="center">
<img src="./assets/0701ELU.png" style="width:70%" />
</p>

