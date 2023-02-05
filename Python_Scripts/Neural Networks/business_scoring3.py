import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn 
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:\\Users\\sakkarkr\\Downloads\\business_score_20200310.csv")

# sampling data
sample = data.sample(n = 10000, replace = False)
sample = sample.drop(['registered','passed_fb','passed_gdn'], axis = 1)

sample['ads_conv'].value_counts()

# Feature Engineering
# Time Elapsed
sample['max_date'] = sample['max_date'].astype(np.datetime64)
sample['min_date'] = sample['min_date'].astype(np.datetime64)
sample['elapse_in_hr'] = (sample['max_date'] - sample['min_date']) / np.timedelta64(1, 'h')

sample = sample.drop(['min_date', 'max_date'], axis = 1)

X = sample.drop('campaign_regist', axis = 1)
y = sample.iloc[:,2]

# Standard Scaler
sc = StandardScaler()
X_sc = sc.fit_transform(X.iloc[:,2:].values)
X_sc = pd.DataFrame(data = X_sc, index = sample.index, columns =  X.iloc[:,2:].columns)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size = 0.33, random_state = 1)


# Building model
# Neural Network
# input shape: 3000 * 25
# hidden shape: 25 * 1
# Output shape: 3000 * 1

class Neural_Network(nn.Module):
  def __init__(self, ):
    super(Neural_Network, self).__init__()
    self.inputSize = 25
    self.outputSize = 1
    self.hiddenSize = 25 # need tuning!!
    
    
    # Weight initialization
    self.W1 = torch.rand(self.inputSize, self.hiddenSize).double()
    self.W2 = torch.rand(self.hiddenSize, self.outputSize).double()

  def forward(self, X):
    self.z = torch.matmul(X, self.W1) 
    self.z2 = self.sigmoid(self.z)
    self.z3 = torch.matmul(self.z2, self.W2)
    o = self.sigmoid(self.z3)
    return o

  def sigmoid(self, s):
    return 1 / (1 + torch.exp(-s))

  def sigmoidPrime(self,s):
    # derivative of sigmoid
    return s * (1 - s)
  
    # fix backward part *****
  def backward(self, X, y, o):
    self.o_error = y - o # error in output
    self.o_delta = self.o_error * self.sigmoidPrime(o) # deriavative of sig to error
    self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
    self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
    self.W1 += torch.matmul(torch.t(X), self.o_delta)
    self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

  def train(self, X, y):
    # fwd + bkwd pass for training
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self, model):
    torch.save(model, "NN")
    # torch.load("NN")

  def predict(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(X_tensor))
    print("Output: \n" + str(self.forward(X_tensor)))

NN = Neural_Network()
X_tensor = torch.tensor(X_sc.values)
X_tensor = X_tensor.double()
y_tensor = torch.tensor(y.values).double()
y_tensor = y_tensor.unsqueeze(1)
for i in range(100):
  print('#' + str(i) + " Loss: " + str(torch.mean((y_tensor - NN(X_tensor)) ** 2).detach().item())) # MSE
  NN.train(X_tensor, y_tensor)
NN.saveWeights(NN)
NN.predict()

# Feedforward neural X(input) -> Hidden -> output with sigmoid activation
X_tensor
W1 = torch.rand(25, 25).double() # (25, 25)
W2 = torch.rand(25, 1).double() # (25, 1)
z = torch.matmul(X_tensor, W1) # (3000, 25) * (25, 25) >> (3000, 25)
z2 = 1 / (1 + torch.exp(-z)) # (3000, 25)
z3 = torch.matmul(z2,W2) # (3000,25) * (25,1)
o = 1 / (1 + torch.exp(-z3)) # (3000, 1)
o_error = y_tensor - o # marked ??
sp = o *  (1 - o)
o_delta = o_error * sp
z2_error = torch.matmul(o_delta, torch.t(W2)) # (3000, 1) * (1 , 25)
sp2 = z2 * (1 - z2) # (3000, 25)
z2_delta = z2_error * sp2  
W1 += torch.matmul(torch.t(X_tensor), z2_delta)
W2 += torch.matmul(torch.t(z2), o_delta)

bce = nn.BCEWithLogitsLoss()
output = bce(o, y_train)
print("\nBCE with Logits Loss\n", output)

# Neural network with relu function
class relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    
    def backward(self, x):
        return self.mask
    

X_tensor
W1 = torch.rand(25, 25).double() # (25, 25)
W2 = torch.rand(25, 1).double() # (25, 1)
z = torch.matmul(X_tensor, W1) # (3000, 25) * (25, 25) >> (3000, 25)
z2 = 1 / (1 + torch.exp(-z)) # (3000, 25)
z3 = torch.matmul(z2,W2) # (3000,25) * (25,1)
o = 1 / (1 + torch.exp(-z3)) # (3000, 1)
o_error = y_tensor - o # marked ??
sp = o *  (1 - o)
o_delta = o_error * sp
z2_error = torch.matmul(o_delta, torch.t(W2)) # (3000, 1) * (1 , 25)
sp2 = z2 * (1 - z2) # (3000, 25)
z2_delta = z2_error * sp2  
W1 += torch.matmul(torch.t(X_tensor), z2_delta)
W2 += torch.matmul(torch.t(z2), o_delta)