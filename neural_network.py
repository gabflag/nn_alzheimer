import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

np.random.seed(123)
torch.manual_seed(123)

#######################################
## Data manipulation and transformation
df = pd.read_csv("xx_projetos/alzheimer/alzheimer_formated.csv")

df_forecasters = df.drop(['0','1','2'], axis=1)
df_classes = df.drop(['3','4','5','6','7','8','9','10','11'], axis=1)

np_array_classes = np.array(df_classes, dtype='float32')

tensor_forecasters = torch.tensor(np.array(df_forecasters), dtype=torch.float)
tensor_classes = torch.tensor(np.array(df_classes), dtype=torch.float)

dataset = torch.utils.data.TensorDataset(tensor_forecasters, tensor_classes)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

#######################################
## Model
class Classifier_torch(nn.Module):
    def __init__(self,):
        super().__init__()

        self.dense_00 = nn.Linear(in_features=9, out_features=4, bias=True)
        self.activation_00 = nn.Sigmoid()

        self.dense_01 = nn.Linear(in_features=4, out_features=4, bias=True)
        self.activation_01 = nn.Sigmoid()

        self.output = nn.Linear(in_features=4, out_features=3, bias=True)
   
    def forward(self, x):
        x = self.dense_00(x)
        x = self.activation_00(x)

        x = self.dense_01(x)
        x = self.activation_01(x)

        x = self.output(x)
        return x

## Traning...
neural_network = Classifier_torch()
erro_function = nn.MSELoss()
optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.001, weight_decay=0.0001)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
neural_network.to(device)

print('\n\n#########')
for epoch in range(20):
    running_loss = 0.
    running_mse_0 = 0.
    running_mse_1 = 0.
    running_mse_2 = 0.
    
    for i, data in enumerate(train_loader):
        forecasters, classes = data
        forecasters, classes = forecasters.to(device), classes.to(device)
        optimizer.zero_grad()

        outputs = neural_network(forecasters)
        value_loss = erro_function(outputs, classes)

        mse_0 = nn.functional.mse_loss(outputs[:,0], classes[:,0]).item()
        mse_1 = nn.functional.mse_loss(outputs[:,1], classes[:,1]).item()
        mse_2 = nn.functional.mse_loss(outputs[:,2], classes[:,2]).item()

        running_mse_0 += mse_0 * len(forecasters)
        running_mse_1 += mse_1 * len(forecasters)
        running_mse_2 += mse_2 * len(forecasters)

        value_loss.backward()
        
        optimizer.step()

        value_loss = value_loss.item()
        running_loss += value_loss * len(forecasters)

    print('Epoch {}, erro_00: {:.4f}, erro_01: {:.4f}, erro_02: {:.4f}, loss: {:.4f}'.format(epoch+1,
                                                                                            running_mse_0/len(dataset),
                                                                                            running_mse_1/len(dataset),
                                                                                            running_mse_2/len(dataset),
                                                                                            running_loss/len(dataset)))
    
# Testing and Accuracy
neural_network.eval()

predictions_for_accuracy = neural_network.forward(tensor_forecasters)
np_array_accuracy = predictions_for_accuracy.detach().numpy()

print('\n\n#########')
print(f"Mean of predicted values: {np_array_accuracy.mean(axis = 0)}")
print(f"Mean of real values: {np_array_classes.mean(axis = 0)}\n") 

# Save model
torch.save(neural_network.state_dict(), 'checkpoint.pth')
