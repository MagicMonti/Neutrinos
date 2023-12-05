import pandas as pd
from Dataset import *
import os
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



















