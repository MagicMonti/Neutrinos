import pandas as pd
from Dataset import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

do_creation = False

if do_creation:
    print("creating dataset")
    df = create_dataset()
    #safe dataframe
    df.to_pickle("simulation_data.pkl")

#read dataframe
print("loading dataset ...")
df = pd.read_pickle("simulation_data.pkl")
print("shuffle dataset")
df = df.sample(frac=1) #shuffle data
df = df.reset_index(drop=True)
print("done!")
print("showing the first entries (precicions is higher than it apperars)")
print(df.head())
print(df.shape)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Sequential(
            nn.Linear(2, 15).double(),
            nn.ReLU(),
            nn.Linear(15, 15).double(),
            nn.ReLU(),
            nn.Linear(15, 1).double(),
        )
        self.C2 = nn.Sequential(
            nn.Linear(2, 15).double(),
            nn.ReLU(),
            nn.Linear(15, 15).double(),
            nn.ReLU(),
            nn.Linear(15, 1).double(),
        )

    def forward(self, x):
        #x[0]...distance
        #x[1]....beta 
        #x[2]....mass
        beta_m = torch.cat((
            x[1].reshape(-1,1),
            x[2].reshape(-1,1)
            ),1)
        
        temp = self.C1(beta_m).reshape(1,-1)
        temp = temp * torch.pow(x[0],2)
        temp = self.C2(beta_m).reshape(1,-1)
        temp = temp * x[0]

        return temp


model = Network().to(device)
print(model)

x = torch.cat(
    (torch.tensor(df["distance"].values).reshape(-1, 1)  ,
    torch.tensor(df["beta"].values).reshape(-1,1), 
    torch.tensor(df["mass"].values).reshape(-1,1))
    ,1).reshape(3,-1)
y = torch.tensor(df["deflection"].values).reshape(1,-1) 



print(x.shape)
print(y.shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




