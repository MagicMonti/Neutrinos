import numpy as np
from trajectory import Geodesic,Correction
import matplotlib.pyplot as plt
from scipy import constants as const
import pandas as pd


def create_data_set(beta_list, m_list):
    c = const.c

    M = 5.972E24 
    r_s = 2* const.G * M/c**2
    hbar = 1
    l_p = 1.9732705E-7
    m_p = 1.7826627E-36 


    #r_s in natrual units
    r_s = r_s/l_p

    r_start = 6371000 /l_p #earth radius in natural units

    

    x_0 = np.array([0, r_start, np.pi/2, 0], dtype=np.double)


    simulation_length = 50000
    d_tau = 10131712


    dataframes = []


    for beta in beta_list:
        geodesic = Geodesic(beta, x_0, r_s, r_start)
        x_geo,u_geo = geodesic.get_path(simulation_length=simulation_length, dtau = d_tau)
        for m in m_list:
            corr = Correction(x_0,m,r_s, x_geo, u_geo)
            x,_ = corr.get_path(simulation_length=simulation_length, dtau = d_tau)
            x = x.T

            y=x[1]*np.sin(x[2])*np.sin(x[3])
            x=x[1]*np.sin(x[2])*np.cos(x[3])

            x_geodesic=x_geo.T[1]*np.sin(x_geo.T[2])*np.cos(x_geo.T[3])


            df = pd.DataFrame({
                "distance" : y,
                "deflection" : x-x_geodesic,
                "beta" : np.ones(simulation_length+1)*beta,
                "mass" : np.ones(simulation_length+1)*m
            })
            print(m)
            df.distance = df.distance.astype(np.double)
            df.deflection = df.deflection.astype(np.double)
            df.beta = df.beta.astype(np.double)
            df.mass = df.mass.astype(np.double)
            dataframes.append(df)

    return pd.concat(dataframes)



def create_training_dataset():
    m_p = 1.7826627E-36 
    m = 2.14E-37 
    m = m / m_p
    beta_list = np.array([0.99,0.998, 0.999])
    m_list = np.array([m*1E-7 *2.5, m*1E-7 *2, m*1E-7 *1.5, m*1E-7 *1, ], dtype=np.double)
    return create_data_set(beta_list, m_list)

def create_test_dataset():
    m_p = 1.7826627E-36 
    m = 2.14E-37 
    m = m / m_p
    beta_list = np.array([0.9999])
    m_list = np.array([m*1E-7 *2.1], dtype=np.double)
    return create_data_set(beta_list, m_list)


    

        





