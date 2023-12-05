import pandas as pd
from Dataset import *
import os
<<<<<<< HEAD
#from torchvision import datasets, transforms
#import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from matplotlib import cm

l_p = 1.9732705E-7

do_creation = False
calc_params = True

if do_creation:
    print("creating dataset")
    df = create_training_dataset()
    df.to_pickle("simulation_data.pkl")

    print("creating dataset")
    df = create_test_dataset()
    df.to_pickle("validation_data.pkl")

#read dataframe
print("loading dataset ...")
df = pd.read_pickle("simulation_data.pkl")
df_validation = pd.read_pickle("validation_data.pkl")




def deflection_func(distance, C_1, C_2):
    return C_1*distance**2 + C_2*distance


def C2_func(data, A, B, C,D):
    beta = data[0]
    m = data[1]

    #return A*np.exp(B*m + C*(1/(1-beta)))
    return A*np.exp(-B*m + C*beta)
    #return (1+A*m +D*m**2)*(B*beta + C)
    #return A*m *(B*beta + C)

def C1_func(data, A, B, C,D):
    beta = data[0]
    m = data[1]

    #return A*(np.exp(B*m + C*beta))
    return np.exp(-A*m)*(B*beta + C)
    #return (1+A*m +D*m**2)*(B*beta + C)
    #return (1+B*m**2 + A*m**3)*(C*beta + D)
    #return *(B*beta + C)

m_list = df["mass"].unique()
beta_list = df["beta"].unique()



print("fitting C1 and C2 ")
C1 = np.zeros((len(m_list),len(beta_list)))
C2 = np.zeros((len(m_list),len(beta_list)))


masses = np.zeros(len(m_list)*len(beta_list))
betas = np.zeros(len(m_list)*len(beta_list))


for i,m in enumerate(m_list):
    df_ = df.loc[df['mass'] == m]
    masses[i*len(beta_list):len(beta_list)*(i+1)] = m*len(beta_list)
    for j,beta in enumerate(beta_list):
        betas[i*len(beta_list)+j] = beta
        df__ = df_.loc[df_['beta'] == beta]
        distance = df__["distance"]
        deflection = df__["deflection"]
        coeffs,_ = curve_fit(deflection_func,distance,deflection)
        C1[i][j] = coeffs[0]
        C2[i][j] = coeffs[1]


print("plotting C1 with as a function of m")
for i in range(len(beta_list)):
    plt.plot(m_list,C1.T[i], label="beta="+str(beta_list[i]))
plt.legend()
plt.savefig("plots/coeffs/C1(m).png")
plt.clf()
print("plotting C2 with as a function of m")
for i in range(len(beta_list)):
    plt.plot(m_list,C2.T[i], label="beta="+str(beta_list[i]))
plt.legend()
plt.savefig("plots/coeffs/C2(m).png")
plt.clf()

print("plotting C1 with as a function of beta")
for i in range(len(m_list)):
    plt.plot(beta_list,C1[i], label="m="+str(m_list[i]))
plt.legend()
plt.savefig("plots/coeffs/C1(beta).png")
plt.clf()
print("plotting C2 with as a function of beta")
for i in range(len(m_list)):
    plt.plot(beta_list,C2[i], label="m="+str(m_list[i]))
plt.legend()
plt.savefig("plots/coeffs/C2(beta).png")
plt.clf()



C1 = C1.flatten()
C2 = C2.flatten()

coeffs_1,_ = curve_fit(C1_func, [betas, masses], C1)
coeffs_2,_ = curve_fit(C2_func, [betas, masses], C2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(masses, betas, C1_func([betas,masses], coeffs_1[0], coeffs_1[1], coeffs_1[2], coeffs_1[3]), label="fitt")
ax.scatter(masses, betas, C1, label="real coefficents")
ax.legend()
plt.savefig("plots/coeffs/3d_C1.png")

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(masses, betas, C2_func([betas,masses], coeffs_2[0], coeffs_2[1], coeffs_2[2], coeffs_2[3]), label="fitt")
ax.scatter(masses, betas, C2, label="real coefficents")
ax.legend()
plt.savefig("plots/coeffs/3d_C2.png")
plt.clf()



def get_defelction(distance, m, beta):

    C1 = C1_func(np.array([beta,m]), coeffs_1[0], coeffs_1[1], coeffs_1[2], coeffs_1[3])
    C2 = C2_func(np.array([beta,m]), coeffs_2[0], coeffs_2[1], coeffs_2[2], coeffs_2[3])
    return deflection_func(distance, C1/3, C2/3)


m_list = df_validation["mass"].unique()
beta_list = df_validation["beta"].unique()

for i,m in enumerate(m_list):
    df_ = df_validation.loc[df_validation['mass'] == m]
    for j,beta in enumerate(beta_list):
        df__ = df_.loc[df_['beta'] == beta]
        distance = np.array(df__["distance"])
        deflection = np.array(df__["deflection"])
        plt.plot(distance*(l_p/1000), deflection*(l_p/1000), label="real ("+str(beta)+","+str(m)+")")
        plt.plot(distance*(l_p/1000), get_defelction(distance, m, beta)*(l_p/1000), label="prediction ("+str(beta)+","+str(m)+")")
plt.legend()
plt.savefig("plots/results/plot.png")
plt.clf()















=======
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
>>>>>>> a4c080d (added neural-network)




