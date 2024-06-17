# Tutorial

1. Tensor  
2. Dataset,DataLoader  
   1. `torch.utils.data`
3. **Transform**  
4. nn-model  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  

## Transform
> 언제나 데이터가 완성된 상태로 올 수는 없다.

모든 TorchVison 데이터셋은 feature를 조작하기 위한 `transform`, label을 조작하기 위한 `target_transform` 파라미터를 가지고 있다.

[`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html) 패키지에 자주 쓰이는 transform 들이 담겨있다.

### FashionMNIST 예시

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,

    # DataLoader 가 아닌 dataset 에 대한 파라미터임에 주의
    # Features
    transform=ToTensor(),
    # labels
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```
#### ToTensor()

- PIL image, NumPy ndarray 를 FloatTensor 로 변형해줌.  

- 값을 [0., 1.] 으로 정규화까지 시켜줌

#### Lambda Transforms

- 사용자가 정의한 어떤 함수도 적용 가능함

- 제시한 예시는 범주형 label (int 형)을 one-hot encoding 형식의 tensor 로 바꿔줌

```
// label
[0,1,0,2,5]

// target_transform
[1,0,0,0,0,0]
[0,1,0,0,0,0]
[1,0,0,0,0,0]
[0,0,1,0,0,0]
[0,0,0,0,0,1]
```
```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

### `torch.Tensor.scatter_`
params:
- `dim` : `0` 한 `열`씩 읽음 / `1` 한 `행`씩 읽음
- `index` : 어떤 index (결과물에서) 에 src를 뿌릴 건지
- `src` : 원본 배열
- `reduce` : `add` : 해당 index에 src를 더해줌 / `multiply` : 곱해줌
```python
>>> src = torch.arange(1, 11).reshape((2, 5))
# tensor([[ 1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10]])

>>> index = torch.tensor([[0, 1, 2, 0]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
#tensor([[1, 0, 0, 4, 0],
#        [0, 2, 0, 0, 0],
#        [0, 0, 3, 0, 0]])

>>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
# tensor([[1, 2, 3, 0, 0],
#         [6, 7, 0, 0, 8],
#         [0, 0, 0, 0, 0]])

>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='multiply')
# tensor([[2.0000, 2.0000, 2.4600, 2.0000],
#         [2.0000, 2.0000, 2.0000, 2.4600]])

>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='add')
# tensor([[2.0000, 2.0000, 3.2300, 2.0000],
#         [2.0000, 2.0000, 2.0000, 3.2300]])
```
