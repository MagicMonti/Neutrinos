import pandas as pd
from Dataset import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
import pickle
import matplotlib.pyplot as plt

do_creation = True
do_training = True

if do_creation:
    print("creating dataset")
    df = create_training_dataset()
    #drop 80% of data
    df = df.sample(frac=0.2)
    #safe dataframe
    df.to_pickle("simulation_data.pkl")

    print("creating dataset")
    df = create_test_dataset()
    #drop 80% of data
    df = df.sample(frac=0.2)
    #safe dataframe
    df.to_pickle("validation_data.pkl")

#read dataframe
print("loading dataset ...")
df = pd.read_pickle("simulation_data.pkl")
df_validation = pd.read_pickle("validation_data.pkl")
print("shuffle dataset")
df = df.sample(frac=1) #shuffle data
df = df.reset_index(drop=True)
print("done!")
print("showing the first entries (precicions is higher than it apperars)")
print(df.head())
print(df.shape)
print("validation....")
print(df_validation.head())
print(df_validation.shape)

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

device = "cpu"

print(f"Using {device} device")


def createBatches(x,y,batchsize = 128):

    print("shapex fo x,y")
    print(x.shape)
    print(y.shape)

    number_of_batches = x.shape[1]/batchsize
    print(number_of_batches)

    splited_x = list(torch.tensor_split(x, int(number_of_batches)+1, dim=1))
    splited_y = list(torch.tensor_split(y, int(number_of_batches)+1, dim=1))

    print("shape of splited x")
    print(len(splited_x))
    print(splited_x[0].shape)

    print("shape of splited y")
    print(len(splited_y))
    print(splited_y[0].shape)

    return (splited_x,splited_y)


def data_normalization(x,y):

    distance_max = torch.max(x[0])
    distance_min = torch.min(x[0])

    beta_max = torch.max(x[1])
    beta_min = torch.min(x[1])

    mass_max = torch.max(x[2])
    mass_min = torch.min(x[2])

    deflection_max = torch.max(y[0])
    deflection_min = torch.min(y[0])


    min_and_max = {
        "distance_max" : distance_max,
        "distance_min" : distance_min,
        "beta_max" : beta_max,
        "beta_min" : beta_min,
        "mass_max" : mass_max,
        "mass_min" : mass_min,
        "deflection_max" : deflection_max,
        "deflection_min" : deflection_min
    }

   

    x[0] = (x[0] - distance_min)/(distance_max - distance_min)
    x[1] = (x[1] - beta_min)/(beta_max- beta_min)
    x[2] = (x[2] - mass_min)/(mass_max - mass_min)
    y[0] = (y[0] - deflection_min)/(deflection_max - deflection_min) 

    print("normalized min and max")
    print(torch.max(x[0]), torch.min(x[0]))
    print(torch.max(x[1]), torch.min(x[1]))
    print(torch.max(x[2]), torch.min(x[2]))
    print(torch.max(y[0]), torch.min(y[0]))



    return (x,y,min_and_max)


def normalize_with_params(x,y,min_and_max):

    distance_max = min_and_max["distance_max"]
    distance_min = min_and_max["distance_min"]
    beta_max = min_and_max["beta_max"]
    beta_min = min_and_max["beta_min"]
    mass_max = min_and_max["mass_max"]
    mass_min = min_and_max["mass_min"]
    deflection_max = min_and_max["deflection_max"]
    deflection_min = min_and_max["deflection_min"]


    x[0] = (x[0] - distance_min)/(distance_max - distance_min)
    x[1] = (x[1] - beta_min)/(beta_max- beta_min)
    x[2] = (x[2] - mass_min)/(mass_max - mass_min)
    y[0] = (y[0] - deflection_min)/(deflection_max - deflection_min) 

    return (x,y)

def reverse_normalization(x,y,min_and_max):

    distance_max = min_and_max["distance_max"]
    distance_min = min_and_max["distance_min"]
    beta_max = min_and_max["beta_max"]
    beta_min = min_and_max["beta_min"]
    mass_max = min_and_max["mass_max"]
    mass_min = min_and_max["mass_min"]
    deflection_max = min_and_max["deflection_max"]
    deflection_min = min_and_max["deflection_min"]

    x[0] = (distance_max - distance_min) * x[0] + distance_min
    x[1] = (beta_max - beta_min) * x[1] + beta_min
    x[2] = (mass_max - mass_min) * x[2] + mass_min
    y[0] = (deflection_max - deflection_min) * y[0] + deflection_min

    return (x,y)





class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Sequential(
            nn.Linear(2, 15).double(),
            nn.ReLU(),
            nn.Linear(15, 15).double(),
            nn.ReLU(),
            nn.Linear(15, 15).double(),
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
            nn.Linear(15, 15).double(),
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





def train(splited_x,splited_y, model, loss_fn, optimizer):

    model.train()

    #loss_value = 0
    for batch_index , (batch_x, batch_y) in enumerate(zip(splited_x, splited_y)):
        batch_y, batch_y = batch_x.to(device), batch_y.to(device)

        y_prediction = model(batch_x)
        loss = loss_fn(y_prediction, batch_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch_index % 1000 == 0:
        #     loss, current = loss.item(), (batch_index + 1) * len(batch_x)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{len(splited_x):>5d}]")
        #loss_value += loss.item()
    #print("avg loss", loss_value/len(splited_y))


if do_training:

    x = torch.cat(
        (torch.tensor(df["distance"].values).reshape(-1, 1)  ,
        torch.tensor(df["beta"].values).reshape(-1,1), 
        torch.tensor(df["mass"].values).reshape(-1,1))
        ,0).reshape(3,-1)

    y = torch.tensor(df["deflection"].values).reshape(1,-1)

    x_validation = torch.cat(
        (torch.tensor(df_validation["distance"].values).reshape(-1, 1)  ,
        torch.tensor(df_validation["beta"].values).reshape(-1,1), 
        torch.tensor(df_validation["mass"].values).reshape(-1,1))
        ,0).reshape(3,-1)  

    y_validation = torch.tensor(df_validation["deflection"].values).reshape(1,-1)

    (normalized_x, normalized_y, min_and_max) = data_normalization(x, y)

    print(min_and_max)

    filehandler = open("min_and_max.obj","wb")
    pickle.dump(min_and_max,filehandler)
    filehandler.close()



    (normalized_x_validation, normalized_y_validation) = normalize_with_params(x_validation, y_validation, min_and_max)

    splited_x, splited_y = createBatches(normalized_x, normalized_y, batchsize=2048)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    epoch = 1
    previous_loss = 100
    while True:
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(splited_x, splited_y, model, loss_fn, optimizer)
        
        
        #print(torch.max(normalized_y_validation))
        #print(torch.min(normalized_y_validation))
        noramlized_y_prediction = model(normalized_x)
        noramlized_y_prediction_validation = model(normalized_x_validation)
        #print(torch.max(noramlized_y_prediction_validation))
        #print(torch.min(noramlized_y_prediction_validation))
        loss = loss_fn(noramlized_y_prediction_validation, normalized_y_validation)
        loss_2 = loss_fn(noramlized_y_prediction, normalized_y)
        print("training loss:", loss_2.item()/len(normalized_y[0]))
        print("validation loss:", loss.item()/len(normalized_y_validation[0]))

        epoch += 1
        if epoch > 10:
            if (previous_loss - loss) < 0:
                torch.save(model.state_dict(), "/home/julian/Documents/Masterarbeit/code/Simulator/model")
                break
        previous_loss = loss

    
#load model
model = Network().to(device)
model.load_state_dict(torch.load("/home/julian/Documents/Masterarbeit/code/Simulator/model"))

m_p = 1.7826627E-36 
m = 2.14E-37 
m = m / m_p

df = df.loc[df['beta'] == 0.9]
df = df.loc[df['mass'] == m*1E-5 *4]

print(df.head())

filehandler = open("min_and_max.obj",'rb')
min_and_max = pickle.load(filehandler)
filehandler.close()


x = torch.cat(
        (torch.tensor(df["distance"].values).reshape(-1, 1)  ,
        torch.tensor(df["beta"].values).reshape(-1,1), 
        torch.tensor(df["mass"].values).reshape(-1,1))
        ,0).reshape(3,-1)

y = torch.tensor(df["deflection"].values).reshape(1,-1)

(normalized_x, normalized_y) = normalize_with_params(x, y, min_and_max)

prediction_y_normalized = model(normalized_x)

(prediction_y,_) = reverse_normalization(normalized_x, prediction_y_normalized, min_and_max)


plt.plot(x[0], y[0], label="real values")
plt.plot(x[0], prediction_y[0], label="prediction")

plt.legend()

plt.savefig("plot.png")













        


