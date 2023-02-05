# Binary clssification
import timeit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

start = timeit.default_timer()

df = pd.read_csv("C:\\Users\\sakkarkr\\Downloads\\business_score_20200310.csv")
df = df.drop(['registered','passed_fb','passed_gdn'], axis = 1)

stop = timeit.default_timer()
print('Time: ', stop - start)  

df.describe()
# class distribution
plot = sns.countplot(x = 'campaign_regist', data = df)
for p in plot.patches:
    plot.annotate(format(p.get_height(),'d'), (p.get_x() + p.get_width() / 2., p.get_height()),ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

""" Encode Ouput Class

df['campaign_regist']  = df['campaign_regist'].astype('category')

encode_map = {'Yes' : 1, 'No' : 0}

df['campaign_regist'].replace(encode_map, inplace = True) """

# Time Elapsed
df['max_date'] = df['max_date'].astype(np.datetime64)
df['min_date'] = df['min_date'].astype(np.datetime64)
df['elapse_in_hr'] = (df['max_date'] - df['min_date']) / np.timedelta64(1, 'h')

df = df.drop(['min_date', 'max_date'], axis = 1)

X = df.iloc[:,3:]
y = df.iloc[:,2]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 99)

# standard scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Model parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# ** custom dataloaders
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def __len__ (self):
        return len(self.X_data)
    
train_data = trainData(torch.FloatTensor(X_train),
                       torch.FloatTensor(y_train))

# test data
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
    
    def __len__ (self):
        return len(self.X_data)
    
test_data = testData(torch.FloatTensor(X_test))

train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE,shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = 1)

# Neural Net Architecture
class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        
        # no. input features is 25
        self.layer_1 = nn.Linear(25, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

# check GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = binaryClassification()
model.to(device)

print(model)

criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logist loss
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE) # interchangeable w/ Stochastic Gradient Descent  

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc* 100)
    
    return acc

accuracy_stats = {'train' : []}
loss_stats = {'train': []}

# Train the model
model.train()
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    loss_stats['train'].append(epoch_loss / len(train_loader))
    accuracy_stats['train'].append(epoch_acc / len(train_loader))
    
    print(f'Epoch {e+0:03}: | Loss:{epoch_loss/len(train_loader):.5f} | Acc:{epoch_acc/len(train_loader):.3f}')

# Visualize Loss and Accuracy
# Create dataframes
train_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars = ['index']).rename(columns = {'index':'epochs'})

train_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars = ['index']).rename(columns = {'index':'epochs'})

# Plot the dataframes
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 7))

sns.lineplot(data = train_acc_df, x = 'epochs', y = 'value',
             hue = 'variable', ax = axes[0]).set_title('Train-Val Accuracy/Epoch')

sns.lineplot(data = train_loss_df, x = 'epochs', y = 'value',
             hue = 'variable', ax = axes[1]).set_title('Train-Val Loss/Epoch')

# Test the model
y_pred_list = []

model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        
# Confusion Matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))
sns.heatmap(confusion_matrix_df, annot = True)

# Classification Report
print(classification_report(y_test, y_pred_list))

# Next step : Weighted Sampler
# Hyperparameter, CrossValidation, Activation Function, Undersampling
