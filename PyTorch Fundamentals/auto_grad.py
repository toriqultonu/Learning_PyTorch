import torch

x = torch.randn(3, requires_grad=True)
print(x)
y = x + 2
print(y)
# z = y * y * 2
# z = z.mean()
# print(z)
y.mean()
y.backward()

print(x.grad)

## three ways to remove grad..
# x.requires_grad_(False)
# x.detach()
# with.torch.no_grad()
