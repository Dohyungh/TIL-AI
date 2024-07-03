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
- `nn.ELU` : Applies the Exponential Linear Unit (ELU) function, element-wise. *
- `nn.Hardshrink` : Applies the Hard Shrinkage (Hardshrink) function element-wise. *
- `nn.Hardsigmoid` : Applies the Hardsigmoid function element-wise. *
- `nn.Hardtanh` : Applies the HardTanh function element-wise. *
- `nn.Hardswish` : Applies the Hardswish function, element-wise. *
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
- `nn.Tanh` : Applies the Hyperbolic Tangent (Tanh) function element-wise. *
- `nn.Tanhshrink` : Applies the element-wise Tanhshrink function. *
- `nn.Threshold` : Thresholds each element of the input Tensor.
- `nn.GLU` : Applies the gated linear unit function.

## Non-linear Activations (other)

- `nn.Softmin` : Applies the Softmin function to an n-dimensional input Tensor.
- `nn.Softmax` : Applies the Softmax function to an n-dimensional input Tensor.
- `nn.Softmax2d` : Applies SoftMax over features to each spatial location.
- `nn.LogSoftmax` : Applies the log(Softmax(𝑥)) function to an n-dimensional input Tensor.
- `nn.AdaptiveLogSoftmaxWithLoss` : Efficient softmax approximation.

[Survey of Activation Functions](https://neverabandon.tistory.com/8)

[wikidocs.net](https://wikidocs.net/60683)

[기울기 소실(Vanishing Gradient)의 의미와 해결방법](https://heytech.tistory.com/388)

[Gradient of ReLu at 0](https://discuss.pytorch.org/t/gradient-of-relu-at-0/64345/4)  

[Gradients for non-differentiable functions](https://pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions)

[Natural Gradient를 위해 보면 좋을 글](https://rlwithme.tistory.com/5)

## 왜 Non-linear Activations 를 사용하는가?

Linear Activation의 경우 단순한 행렬 곱인데, 이는 단순히 괄호를 풀어 계산할 경우에 단 한 개의 Activation function으로 대체가 가능하다는 의미이다. 이는 많은 층을 추가해가면서 각 층별로 활성함수가 적용되어 node들 간의 촘촘한 관계를 구현하겠다는 신경망 구현 방식과 맞지 않다.

> 다만, 선형 활성함수를 사용한 층을 '은닉층'(비선형 활성함수를 사용하는) 과 구별해 '선형층'이라는 별개의 이름으로 부르기도 할 정도로 그 자체로 학습할 수 있는 weight이 생겨난다는 측면에서 의미가 아주 없지는 않다. [wikidocs.net](https://wikidocs.net/60683)

## Vanishing Gradient

[기울기 소실(Vanishing Gradient)의 의미와 해결방법](https://heytech.tistory.com/388)

기울기 소실이란 역전파를 이용한 weight 조정 과정에서 Chain Rule에 따라 계속해서 gradient를 곱하게 되는데, 이 gradient 값이 계속해서 작아져 입력층에 가까워졌을 때 쯤엔 0에 근사해지는 것을 의미한다.

### Sigmoid 함수

$$
S(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x + 1}
$$

<p align="center">
<img src="./assets/0702sigmoid.png" style="width:35%" />
</p>


$$
S'(x) = S(x)(1-S(x))
$$

미분 계산의 편의성과 뉴런의 신경신호 전달 방식을 매우 가깝게 구현한다는 점에서 널리 사용됐던 Sigmoid 함수.
그 미분은 $S(x) = 0.5$ 에서 최댓값 0.25를 가진다. 따라서, 모든 $x$의 범위 내에서 최대 0.25 , 최소 0의 값을 가지므로, 이 미분값(gradient)를 계속해서 곱하다 보면, 그 gradient는 극히 작아질 수 밖에 없다.

> 이에 더해, exp(x)의 계산을 근사치로 수행해야 하는 컴퓨터의 계산오차도 역시 계속해서 가산된다.

#### Bias Shift(편향 이동)
Sigmoid 함수의 Vanishing gradient를 가속시키는 이유가 하나 더 있는데, 바로 **Bias Shift** (편향 이동) 이다.
Sigmoid 함수의 평균은 0이 아닌 0.25인데, 이는 즉 입력값의 합보다 출력층의 합이 더 커질 확률이 높다는 의미이다. 이것은 다시, gradient의 분포가 gradient의 최댓값인 0.25를 가지는 중간부보다는 양 극단에서 관찰될 확률이 점점 커진다는 뜻이다.

### $\tanh$ 함수

$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

<p align="center">
<img src="./assets/0702tanh.png" style="width:35%" />
</p>

$$
f'(x) = 1-f(x)^2
$$

쌍곡 탄젠트 함수는 그 형태가 sigmoid와 매우 유사하지만, 최댓값 1을 가져 sigmoid보다 훨씬 넉넉한 gradient범위를 가졌다. 그러나, 여전히 곱할 수록 작아지는 특성을 벗어나지는 못했다. 다만, Bias Shift 현상은 평균이 0이기 때문에 나타나지 않는다.

> 자연어 처리, 음성인식 을 위한 **recurrent neural networks에서 대부분 사용됨**

### ReLU (Rectified Linear Unit) 함수

$$
f(x) = max(0, x)
$$
<p align="center">
<img src="./assets/0702ReLU.png" style="width:35%" />
</p>

0보다 큰 범위에서 기울기 1, 작은 범위에서 0의 기울기를 가진다. 

> 0에서는 0을 대입하는 것이 일반적인 듯 하다.   
[Gradient of ReLu at 0](https://discuss.pytorch.org/t/gradient-of-relu-at-0/64345/4)  
[Gradients for non-differentiable functions](https://pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions)

따라서, vanishing gradient 문제는 해결되었지만, 활성함수를 통과하지 못한 노드는 0이 곱해져 다시는 활성화 되지 못한다는 (Dying ReLU) 단점이 있습니다.

### Leaky ReLU 함수

$$
LeakyReLU(x) = max(0,x) + negative\_slope * min(0,x)
$$

<p align="center">
<img src="./assets/0702LeakyReLu.png" style="width:35%" />
</p>

ReLU 함수의 단점을 보완하기 위해 음의 입력이 들어왔을 떄, 0이 아닌 매우 작은 기울기를 곱한다. 이로써, 뉴런이 죽는 것을 예방한다.

---

[Survey of Activation Functions](https://neverabandon.tistory.com/8) : 각종 활성함수들에 대해 잘 정리해 놓은 글


<p align="center">
<img src="./assets/0702FamilyOfActivationFunctions.png" style="width:60%" />
</p>

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

$$
ELU'(x) = \begin{cases}
1 &\text{if } x \gt0 \\
f(x) + \alpha &\text{if } x\leq 0
\end{cases}
$$

> ReLU 함수를 부드럽게 깎은 함수, $\alpha$는 대체로 1

### ELU, LReLU, ReLU, SReLU 비교
<p align="center">
<img src="./assets/0701ELU.png" style="width:35%" />
</p>

### 특징
- ReLU와 LReLU가 그랬던 것처럼, 양수 input을 받았을 때 기울기가 1이어서 vanishing gradinet를 예방한다.
- ReLU가 음수 input에 대해서 단순히 0의 값을 취하는 것에 반해 ELU는 음수의 값을 취한다. 이는 Mean activations을 0으로 (일종의 편향이동 관점) 밀어줌으로써 gradient가 natural gradient(분포의 관점에서의 gradient) 에 더 가까워 지게 해 더 빠른 학습이 가능하도록 한다.
> - negative 체제(regime)에서 분명한 saturation 고원(plateau)을 가지고 있기 때문에, 보다 강건한 표현(representations)을 학습할 수 있음. 특히 5개 layer 이상의 특정한 network 구조를 가진 ReLU 및 LReLU에 비해 좀 더 빠른 학습 속도와 좀 더 나은 일반화(generalization)를 제공함. 또한, ReLU의 변종들에 비해 state-of-the-art 결과를 보장함.(??????) [출처](https://neverabandon.tistory.com/8)

[Natural Gradient를 위해 보면 좋을 글](https://rlwithme.tistory.com/5)


## Hard Shrink function

### definition

$$
HardShrink(x) = \begin{cases}
   x &\text{if } x\gt\lambda \\
   x &\text{if } x\lt-\lambda \\
   0 &\text{otherwise}
\end{cases}
$$


<p align="center">
<img src="./assets/0703HardShrink.png" style="width:35%" />
</p>

## Soft Shrink function

### definition

$$
SoftShrinkage(x) = \begin{cases}
   x-\lambda &\text{if } x\gt\lambda \\
   x+\lambda &\text{if } x\lt-\lambda \\
   0 &\text{otherwise}
\end{cases}
$$

<p align="center">
<img src="./assets/0703SoftShrink.png" style="width:35%" />
</p>


## Tanhshrink function

### definition

$$
Tanhshrink(x) = x - tanh(x)
$$

<p align="center">
<img src="./assets/0703TanhShrink.png" style="width:35%" />
</p>
---

> TanhShrink와 HardShrink는 잠재변수 값을 계산하기 위한 희소코딩외에는 거의 사용되지 않는다.  
[Shrink activation function Usage](https://deeesp.github.io/deep%20learning/DL-Activation-Functions/#hardshrink---nnhardshrink)
>> **잠재변수(Latent Variable)** : 신호를 이루는 기저 신호들, 직접 측정할 수 없어 통계적 방법론으로 추정함.
>> **희소코딩** : 잠재변수를 알아내기 위해 알고리즘을 돌리는 것(?) 그 결과가 희소 행렬 형태로 나온다(?)  
[희소코딩?](https://wordbe.tistory.com/134)
>> $$x=Da \\ \text{ where } D = (d_1d_2...d_m)$$
>> x : 영상(신호)  
>> $D$ : 사전(dictionary) (희소행렬)  
>> $a$ : 희소코드 (계수집합)  
>> $d_i$ : 사전요소


## Hard Sigmoid fuction

### definition

$$
Hardsigmoid(x) = \begin{cases}
0 &\text{if } x\leq -3,\\
1 &\text{if } x \geq 3,\\
x/6 + 1/2 &\text{otherwise}
\end{cases}
$$

<p align="center">
<img src="./assets/0703HardSigmoid.png" style="width:35%" />
</p>

[BinaryConnect Training Deep Neural Networks with binary weights during propagations](./assets/0703BinaryConnect_Training%20Deep%20Neural%20Networks%20with.pdf) 에서 처음 도입됨

- binary weights(-1, 1) 만으로 빠르고 저렴하게 심층 신경망을 만들고 싶을 떄 사용하면 아주 좋다.
- software 적으로도 그렇고, 어떤 hardware 가속기를 사용하든지 상관 없이 좋은 성능을 보였다고 한다.


## Hard Tanh function

### definition

$$
HardTanh(x) = \begin{cases}
max\_val &\text{if } x\gt max\_val,\\
min\_val &\text{if } x \lt min\_val,\\
x &\text{otherwise}
\end{cases}
$$

<p align="center">
<img src="./assets/0703HardTanh.png" style="width:35%" />
</p>

> Tanh function의 계산적인 효율성을 고려한 버전임.   
> 자연어 처리에서 성공적으로 적용 되었음 [출처](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

## Hard Swish(SiLU) function

### definition

$$
Hardswish(x) = \begin{cases}
0 &\text{if } x\leq -3,\\
x &\text{if } x \geq 3,\\
x \cdot (x+3)/6 &\text{otherwise}
\end{cases}
$$

<p align="center">
<img src="./assets/0703HardSwish.png" style="width:35%" />
</p>

> 신경망에서 더 깊게 진행함에 따라 nonlinearity 를 적용하는 비용은 점점 줄어든다. 해상도(픽셀수)가 다음 층으로 넘어갈 떄마다 대부분 절반으로 떨어지기 떄문이다. `swish` 모델 역시 망의 깊은 부분에서 효과적이라는 것을 알고 있기 때문에, 우리 모델에서 `hard-swish` 도 모델의 후반부에서만 사용했다. [출처](0703SearchingforMobileNetV3)