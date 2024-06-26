# Convolution

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
            1. **Convolution**
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---

## Convolution

함수와 함수의 함성곱부터 시작한다.

$$
f(t) * g(t) := \int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau = (f*g)(t)
$$

그대로 해석하면, 모든 정의역에서 $g$함수를 **뒤집어**(i.e. y축에 대해) f 함수와 **곱해 적분**한다는 뜻이다.

하지만, 우리는 이 연산을 행렬에 대해서 수행할 것이기 때문에, 이산 (시간) 합성곱 (Discrete Time Convolution)을 계산하는 방식을 굳이 알아둘 필요가 있다.

$$
x[n]*h[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]
$$

다음의 안 성립하면 안 될 것 같은 것들이 다행히 성립한다.
- 교환법칙
- 결합법칙
- 분배법칙

### 예시

$$
[1,2,3]*[4,5,6]
$$

```python
# 뒤집는다.
[1,2,3]
[6,5,4]

왼쪽에서 오른쪽으로 쭉 옮긴다. (index를 옮기는 것이 결국 배열을 이동시키는 것과 같다.)

[1,2,3]
4]
= 4

[1,2,3]
5,4]
= 5 + 8 = 13

[1,2,3]
[6,5,4]
= 6 + 10 + 12 = 28

[1,2,3]
  [6, 5
= 12 + 15 = 27

[1,2,3]
    [6
= 18


# 정답
[4, 13, 28, 27, 18]
```

## Convolution을 생각하는 다양한 관점

### 주사위

### Signal

## Convolution의 시간 복잡도

$O(N^2)$ 



### Fast Fourier Transform

$O(Nlog(N))$


## Convolution 2D

## CNN