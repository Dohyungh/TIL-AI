# Tutorial

**1. Tensor**  
2. Dataset, DataLoader  
3. Transform  
4. nn-model  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  

## Tensor
- 배열, 행렬과 유사 (특히 Numpy의 ndarray)
- 모델의 입력, 출력과 그 encoding에 사용 
    - `encoding` : 이미지, 오디오 -> nn에 전달 가능한 데이터로 변환
- GPU를 비롯한 Accelerator 에서 사용가능한 ndarray
- 자동미분(Autograd(Pytorch의 자동미분))에 최적화
    - Computational Graph
    - Gradient Tracking


## Tensor 생성

```python
torch.tensor([[1., -1.], [1., -1.]])
torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
```

- 여러 데이터 타입의 tensor 를 만들 수 있는데,
- tensor를 생성할 때는 `torch.empty()` 와 같은 factory function을 사용하면서 `dtype` arg를 지정해주는 것을 추천한다.
- torch.Tensor 생성자는 Float 형식을 기본형으로 한다.
- python `list` 와 같은 열거형 자료형에 대해 `torch.tensor()` 생성자로 tensor를 생성할 수 있다.
> 1. `torch.tensor()` 는 언제나 데이터를 copy 한다.  
> 2. 이미 존재하는 Tensor 의 특성을 바꾸기 위한 함수들, 예를 들어 `requires_grad_()`, `detach()`와 같은 함수들은 copy 없이 작동한다.  
> 3. numpy array를 copy 없이 Tensor로 만들고 싶다면 `torch.as_tensor()` 를 써라.

```python
x_ones = torch.ones_like(x_data) 
x_rand = torch.rand_like(x_data, dtype=torch.float)
```

- `torch.*_like(data)` 의 경우, data의 다른 arg 들을 복사해서 *에 따라 같은 size의 tensor를 만들어준다.
- 즉, `torch.*(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)` 와 동등하다.


**torch.tensor(requires_grad=True)**
- `requires_grad` arg를 True 로 설정해주면, 
- torch.autograd 가 해당 tensor에 가해진 조작을 기억하고, 빠르게 미분할 수 있도록 돕는다.
```python
x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
out = x.pow(2).sum()
out.backward()
x.grad
```
> 내부 코드를 뜯어보다가 @Overload decorator란 걸 보게됐는데, typing library 에서 제공하는 메서드란다.. 진짜 Java의 오버로딩 하듯이 여러 시그니처를 받을 수 있게 한다.
> exponent 에 Tensor가 오면 각 element-wise 거듭제곱을 시켜주고, Number가 오면 모든 element에 대해 동일한 숫자로 거듭제곱 시켜주는 방식을 구현하기 위해 사용됐다.

## Tensor 연산
```python
# 옮길 수 있다면
if torch.cuda.is_available():
    # 옮겨줘
    tensor = tensor.to("cuda")
```
- `.to("cuda")` tensor 옮기기
    - *큰 tensor를 CPU에서 GPU 로 복사하는 것은 아주 메모리, 시간 낭비라는 걸 기억하자.*

### indexing, slicing

```python
tensor[0]
tensor[:, 0] # 첫번째 컬럼
tensor[..., -1] # 마지막 컬럼
tensor[:,1] = 1
```
- numpy 와 비슷한 방법들을 그대로 사용 가능하다.

### joining
```python
print(tensor)
t1 = torch.cat([tensor, tensor,tensor], dim=1)
print(t1)
t2 = torch.stack([tensor,tensor,tensor])
print(t2)
```
```python
# tensor
tensor([[1, 1, 1, 1],
        [1, 1, 1, 1]], dtype=torch.int32)

# cat
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)

# stack
tensor([[[1, 1, 1, 1],
         [1, 1, 1, 1]],

        [[1, 1, 1, 1],
         [1, 1, 1, 1]],

        [[1, 1, 1, 1],
         [1, 1, 1, 1]]], dtype=torch.int32)
```

### Arithmetic Ops
```python
# 행렬곱
# 두 결과는 동일
# A @ B == A.matmul(B)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

# element-wise 곱
# 두 결과는 동일
z1 = tensor * tensor
z2 = tensor.mul(tensor)
```

- `item()` 으로 원소가 하나인 tensor 에서 그 원소를 뽑아올 수 있음

### 덮어쓰기 연산
- 접미사로 `_`를 붙이면 `=`연산자로 대입해주지 않아도 자동으로 tensor 에 대입해줌
- `x.copy_(y)`, `x.t_()`, `x.add_(1)`


## Bridge with NumPy
`numpy array = tensor.numpy()` 와   
`tensor = torch.from_numpy(numpy array)`  
형식에서 Numpy array 와 Torch tensor 는 **같은 메모리 공간을 공유한다!**

`하나를 조작하면 다른 하나도 변한다.`

- `torch.tensor(numpyArray)`의 경우 하나를 복사해 생성하기 때문에 영향을 받지 않는다.
