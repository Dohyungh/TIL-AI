import torch
import numpy as np

tensor = torch.tensor([[1., -1.], [1., -1.]])
tensor = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))

# 0으로 채우거나 1로 채울 수 있음
tensor = torch.zeros([2, 4])
tensor = torch.ones([2, 4], dtype=int)
tensor = torch.ones([2, 4], dtype=torch.int32)

# device를 지정할 수 있음
# cuda0 = torch.device('cuda:0')
# tensor = torch.ones([2, 4], dtype=torch.float64, device=cuda0)

# python에서 배열에 접근하는 것처럼 값을 수정하거나 가져올 수 있음
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(x[1][2])
x[0][1] = 8
# print(x)/

# requires_grad
x = torch.tensor([[2., -2.], [2., 2.]], requires_grad=True)
# 모두 제곱해서 전부 더해라
out = x.pow(2).sum()
# gradient를 계산해라 (없으면 x 에 grad가 none 으로 나온다.)
# 다 제곱 해서 다 더했으니까, - 붙은 원소 빼고 모두 2 * 원래 값 가 나올 것임.
# x^2 -> 2x
out.backward()
x.grad
# tensor([[ 4.0000, -4.0000],
#        [ 4.0000,  4.0000]])


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# print(tensor)
t1 = torch.cat([tensor, tensor,tensor], dim=1)
# print(t1)
t2 = torch.stack([tensor,tensor,tensor])
# print(t2)


t = torch.ones(5)
# print(f"t: {t}")
n = t.numpy()
# print(f"n: {n}")

t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

n = np.ones(5)
# t = torch.from_numpy(n)
t = torch.tensor(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")