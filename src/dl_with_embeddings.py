# %% imports
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import numpy as np
import pandas as pd
import pandarallel
from pandarallel import pandarallel
from sgt import SGT
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
embeddings_joined = pd.read_csv("embeddings_first_20000_Antibodies_kappa_5.csv")
# shuffle frame
embeddings_joined = embeddings_joined.sample(frac=1)
paired_embeddings = embeddings_joined.iloc[:round(embeddings_joined.shape[0]/2), :]
non_paired_embeddings = embeddings_joined.iloc[round(embeddings_joined.shape[0]/2):, :]


#%%
heavy_half = non_paired_embeddings.iloc[:,:401]
light_half_shuffled = non_paired_embeddings.iloc[:,401:].sample(frac=1)

non_paired_embeddings = pd.concat([heavy_half, light_half_shuffled], axis=1).sample(frac=1)
non_paired_embeddings["id"] = ["shuffled_" + str(i) for i in range(len(non_paired_embeddings.index))]
# %%
numerical_columns = embeddings_joined.columns[1:]
# 1 = paired 0 = unpaired
output = [1] * round(embeddings_joined.shape[0]/2) + [0] * round(embeddings_joined.shape[0]/2)


dataset_df = pd.concat([paired_embeddings, non_paired_embeddings], axis=0)
dataset_df["label"] = output
#shuffle dataset df again
dataset_df = dataset_df.sample(frac=1)

# %%
data = np.stack([dataset_df[col].values for col in numerical_columns], 1)
data = torch.tensor(data, dtype=torch.float)
data[:5]

outputs = torch.tensor(dataset_df["label"].values).flatten()
outputs[:5]
# %%
# TRAIN TEST
total_records = len(outputs)
test_records = int(total_records * .2)

train_data = data[:total_records-test_records]
test_data = data[total_records-test_records:total_records]

train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]
# %%
# declare nn class
class Model(nn.Module):

    def __init__(self, num_cols, output_size, layers, p=0.8):
        super().__init__()
        self.batch_norm_num = nn.BatchNorm1d(num_cols)

        all_layers = []
        input_size = num_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.batch_norm_num(x)
        x = self.layers(x)
        return x

#%%
# train
model = Model(data.shape[1], 2, [200,100,50], p=0.4)
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300
aggregated_losses = []

for i in range(epochs):
    i += 1
    y_pred = model(train_data)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss.detach().numpy())

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
# %%
# plot loss function
plt.plot(range(epochs), aggregated_losses)
plt.ylabel('Loss')
plt.xlabel('epoch')

# %% 
# predict

with torch.no_grad():
    y_val = model(test_data)
    loss = loss_function(y_val, test_outputs)
print(f'Loss: {loss:.8f}')
# %%


#### try a cnn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 400, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(200, 100, 2)
        self.fc1 = nn.Linear(100 * 2 * 2, 200)
        self.fc2 = nn.Linear(200, 75)
        self.fc3 = nn.Linear(75, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 100 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

batch_size=100
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, inputs in enumerate(trainloader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = net(inputs[None, ...])
        loss = criterion(y_pred, train_outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
# %%
