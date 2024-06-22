# Tutorial

1. Tensor  
2. Dataset,DataLoader  
   1. `torch.utils.data`
3. Transform
   1. `torchvision.transforms.v2`
      1. `v2.CutMix`
      2. `v2.MixUp`
   2. `torchvision.transforms`
4. **nn-model**  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  

[PyTorch-Tutorial-nnModel](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)를 참고해 작성됨

## Build the Neural Network

NN은 data를 연산하는 layer 들과 module 들로 이루어져 있다. `torch.nn`에 NN을 구성하는 데에 필요한 건 다 있을 것이다.  
PyTorch의 모든 module 들은 `nn.Module`의 하위 클래스인데, 이건 즉 모델 자체가 모듈이면서 모듈을 담고 있다는 뜻이다. 이런 중첩 구조가 복잡한 구조를 쉽게 만드는 원동력이다.

## 예시 : FashionMNIST Image Classifier

### Device (가속기 결정) 정하기

GPU 나 MPS (Metal Performance Shaders)를 써보는 것이 좋을 텐데, 각각 `torch.cuda`와 `torch.backends.mps` 가 이용가능한지 확인해 봐야 한다. 둘다 안된다면 CPU를 사용하자.

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

> MPS?   
> Apple에서 만든 그래픽 API, 예전부터 디자인 하는 사람들이 Apple 제품을 많이 쓴 이유인가 보다.

> M3 macbook air 에서 위 코드를 돌리면 "mps" 로 설정된다. Pro 모델부터 그래픽 카드가 별도로 들어가서 그런건지, Cuda 관련해서 설치한게 아무것도 없어서인지 모르겠지만, 모델은 집에 있는 3070으로 돌릴 거니까.. 일단 넘어가기로 한다. (아마 집에서 예전에 Cuda를 깔았던 기억이 있다. 돌아가겠지?)


### Class 정의하고 사용하기

1. `nn.Module`을 상속하는 class를 만들자.  
2. `__init__`안에 모델의 layer들을 초기화하자. 

3. **모든 `nn.Module`의 하위 클래스는 `forward` 메서드에 데이터에 대한 연산을 구현한다!**

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits     
```

4. 작성한 클래스의 인스턴스를 만들어서 `device`로 옮겨준다.

`model = NeuralNetwork().to(device)` 요렇게 `.to()`메서드를 쓰자.

5. 모델을 사용할 때는 직접 `forward()`함수를 호출하지 말고! 그냥 데이터를 모델에 인자로 넘겨주자. (약간의 background operations 가 있단다.)

모델을 호출하면 2차원의 `[[]]`로 생긴 tensor를 반환하는데, 0 차원에는 각각의 class에 대한 예측값 10개가, 1차원에는 실제 예측값 개개인이 들어있다.
> 무슨 말인지 이해가 안됐는데 그냥 결과값을 보면 이렇다.
>> tensor( [ [ 0.0982, 0.0998, 0.0960, 0.0979, 0.1122, 0.0948, 0.0904, 0.1107, 0.0980, 0.1020 ] ] , device='mps:0', grad_fn=<SoftmaxBackward0>)

이 결과를 `nn.Softmax`에 넘겨주면 예측 클래스를 얻을 수 있다.
(그냥 가장 큰 값을 가진 class를 골라준다는 뜻이다.)

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

## Model Layers

### `nn.Flatten`

2D 이미지를 연속적인 1차원으로 만들어준다. (ex. 28*28 -> 784)

#### args

`start_dim` : 처리할 시작 dim  
`end_dim` : 끝 dim

> `start_dim`의 기본값이 1 이고, `end_dim`의 기본값이 -1 이라서
기본적으로 0 차원은 `batch_size` 임을 가정한다.

### `nn.Linear`
정말 많이 쓰이는 단순 선형 layer.

$$y=xA^T + b$$

#### args

`in_features` : input size
`out_features` : output size
`bias`(bool) : bias(b항) 학습 시킬 거냐 말거냐

bias 가 없으면 그냥 행렬곱이 된다.

### `nn.ReLU`

비선형 활성함수(Activation function)를 쓰셔도 됩니다.
> 뭔가 공식문서에서 권장하는 느낌이 든다.

$$ ReLU(x) = (x)^+= max(0,x) $$
> 0보다 작은 걸 다 0으로 없애 준다.

### `nn.Sequential`
여러 모듈을 하나로 묶어주는 역할을 한다.

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### `nn.Softmax`
`exp(x_i) / SUM(exp(x_i))` 식으로 계산해서 결과의 모든 값을 합하면 확률이라 1이 나온다.

#### args

`arg`가 int 형의 `dim`만 있는데, 사용할 때 조금 헷갈린다.  
softmax를 계산해 나갈 때 따라갈 차원을 의미한다. 즉, `dim=0` 이면, 가장 바깥 대괄호를 따라간다는 의미이다.
각 행을 돌면서 하나씩 값을 따온다. 결과적으로 열의 합이 1로 만들어진다.

`dim=0` : 열의 합이 1로 만들어진다.

`dim=1` : 행의 합이 1로 만들어진다.

## Model Parameters

많은 layer들이 변수화되기 마련인데, 가중치들과 bias 들이 학습중에 최적화된다는 것을 말한다.
`nn.Module`을 상속하면서 model 객체 안에 정의된 모든 필드들이 트래킹되고, 모든 변수들이 `parameters()` 혹은 
`named_parameters()` 메서드를 통해 접근 가능해집니다.

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

```
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)
```