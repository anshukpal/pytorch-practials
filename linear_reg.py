import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)

print(x_values)
x_train = x_train.reshape(-1, 1)
# print(x_train.shape)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

print(y_values)

'''
CREATE MODEL CLASS
'''
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  
    
    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)
# model.cuda()
'''
INSTANTIATE LOSS CLASS
'''

criterion = nn.MSELoss()

'''
INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 100
for epoch in range(epochs):
    epoch += 1
    # Convert numpy array to torch Variable
    
    #######################
    #  USE GPU FOR MODEL  #
    #######################
    # if torch.cuda.is_available():
    inputs = Variable(torch.from_numpy(x_train))
        
    #######################
    #  USE GPU FOR MODEL  #
    #######################
    # if torch.cuda.is_available():
    labels = Variable(torch.from_numpy(y_train))
    # Clear gradients w.r.t. parameters
    optimizer.zero_grad() 
    
    # Forward to get output
    outputs = model(inputs)
    
    # Calculate Loss
    loss = criterion(outputs, labels)
    
    # Getting gradients w.r.t. parameters
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch,loss.data))
#     print('input {}, output {}'.format(inputs,outputs))