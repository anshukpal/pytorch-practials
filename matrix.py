import torch
import numpy as np

arr = [[1,2],[3,4]]
# print(arr)
# print(np.array(arr))
# print(torch.Tensor(arr))

# print(np.ones((2,2)))
# print(torch.ones((2,2)))

# np.random.seed(0)
# print(np.random.rand(2,2))
# torch.manual_seed(0)
# print(torch.rand(2,2))

# print(torch.cuda.is_available())

print(type(np.ones((2,2))))
torch_tensor = torch.from_numpy(np.ones((2,2)))
print(type(torch_tensor))

print(torch.cuda.is_available())
