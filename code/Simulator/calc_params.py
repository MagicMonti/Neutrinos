import pandas as pd
from Dataset import *
import os
#from torchvision import datasets, transforms
#import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


def C_of_m_func(m,A,B):
    return A*np.exp(-B*m)

def func_for_A(beta,E,F):
    return E*np.exp(beta*F)

def func_for_B(beta,E,F):
    return E*beta+ F

m_list = df["mass"].unique()
beta_list = df["beta"].unique()



print("fitting C1 and C2")
C1 = np.zeros((len(m_list),len(beta_list)))
C2 = np.zeros((len(m_list),len(beta_list)))
for i,m in enumerate(m_list):
    df_ = df.loc[df['mass'] == m]
    for j,beta in enumerate(beta_list):
        df__ = df_.loc[df_['beta'] == beta]
        distance = df__["distance"]
        deflection = df__["deflection"]
        plt.plot(distance, deflection)
        coeffs,_ = curve_fit(deflection_func,distance,deflection)
        C1[i][j] = coeffs[0]
        C2[i][j] = coeffs[1]
    plt.savefig("plot_of_different_betas_at_m"+str(m)+".png")
    plt.clf()


A_1 = np.zeros(len(beta_list))
B_1 = np.zeros(len(beta_list))

A_2 = np.zeros(len(beta_list))
B_2 = np.zeros(len(beta_list))

print("fitting A1 and B1")
for i in range(len(beta_list)):
    coeffs,_ = curve_fit(C_of_m_func,np.array(m_list),C1.T[i])
    A_1[i] = coeffs[0]
    B_1[i] = coeffs[1]
    plt.plot(m_list,C1.T[i], label="beta="+str(beta_list[i]))

plt.legend()
plt.savefig("C1.png")
plt.clf()

print("fitting A2 and B2")
for i in range(len(beta_list)):
    coeffs,_ = curve_fit(C_of_m_func,np.array(m_list),C2.T[i])
    A_2[i] = coeffs[0]
    B_2[i] = coeffs[1]
    plt.plot(m_list,C2.T[i], label="beta="+str(beta_list[i]))
plt.legend()
plt.savefig("C2.png")

plt.clf()

plt.plot(beta_list, A_1)
plt.savefig("A_1.png")
plt.clf()
plt.plot(beta_list, A_2)
plt.savefig("A_2.png")
plt.clf()
plt.plot(beta_list, B_1)
plt.savefig("B_1.png")
plt.clf()
plt.plot(beta_list, B_2)
plt.savefig("B_2.png")
plt.clf()

coeffs_for_A_1,_ = curve_fit(func_for_A,np.array(beta_list),A_1)
coeffs_for_A_2,_ = curve_fit(func_for_A,np.array(beta_list),A_2)
coeffs_for_B_1,_ = curve_fit(func_for_A,np.array(beta_list),B_1)
coeffs_for_B_2,_ = curve_fit(func_for_A,np.array(beta_list),B_2)


def get_defelction(distance, m, beta):

    A_1 = func_for_A(beta, coeffs_for_A_1[0], coeffs_for_A_1[1])
    A_2 = func_for_A(beta, coeffs_for_A_2[0], coeffs_for_A_2[1])

    B_1 = func_for_A(beta, coeffs_for_B_1[0], coeffs_for_B_1[1])
    B_2 = func_for_A(beta, coeffs_for_B_2[0], coeffs_for_B_2[1])

    C1 = C_of_m_func(m, A_1, B_1)
    C2 = C_of_m_func(m, A_2, B_2)
    return deflection_func(distance, C1, C2)


m_list = df_validation["mass"].unique()
beta_list = df_validation["beta"].unique()

for m in m_list:
    df_ = df_validation.loc[df_validation['mass'] == m]
    for beta in beta_list:
        df__ = df_.loc[df_['beta'] == beta]
        distance = df__["distance"]
        deflection = df__["deflection"]
        plt.plot(distance, deflection, label="real")
        plt.plot(distance, get_defelction(distance, m, beta), label="prediction")
    plt.legend()
    plt.savefig("plot_of_different_betas_at_m_prediction"+str(m)+".png")
    plt.clf()



















