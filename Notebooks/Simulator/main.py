import numpy as np
from trajectory import Geodesic,Correction
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy.optimize import curve_fit

c = const.c

M = 5.972E24 
r_s = 2* const.G * M/c**2
hbar = 1
l_p = 1.9732705E-7
m_p = 1.7826627E-36 


#r_s in natrual units
r_s = r_s/l_p

r_start = 6371000 /l_p #earth radius in natural units

m = 2.14E-37 

beta = 0.98
m = m / m_p

x_0 = np.array([0, r_start, np.pi/2, 0])


simulation_length = 50000
d_tau = 10131712

geodesic = Geodesic(beta, x_0, r_s, r_start)
x_geo,u_geo = geodesic.get_path(simulation_length=simulation_length, dtau = d_tau)

m_list = np.array([m*1E-5 *4, m*1E-5 *3.5 ,m*1E-5 *3, m*1E-5 *2.5, m*1E-5 *2, m*1E-5 *1])

coef_m = []

x_geo = x_geo.T
y_geo=x_geo[1]*np.sin(x_geo[2])*np.sin(x_geo[3])
x_geo=x_geo[1]*np.sin(x_geo[2])*np.cos(x_geo[3])

print("mass of neutrino ", m)

def deflection(y, C_1, C_2):
    return C_1*y**2 + C_2*y

def C_of_m(m,A,B):
    return A*np.exp(-B*m)

for m in m_list:
    corr = Correction(x_0,m,r_s, x_geo, u_geo)
    x,_ = corr.get_path(simulation_length=simulation_length, dtau = d_tau)
    x = x.T

    y=x[1]*np.sin(x[2])*np.sin(x[3])
    x=x[1]*np.sin(x[2])*np.cos(x[3])

    plt.plot(y*l_p/1000,(x-x_geo)*l_p/1000, label="m="+str(m))
    coeffs, _ = curve_fit(deflection,y, (x-x_geo))
    coef_m.append(coeffs)


#plt.plot(y_geo,x_geo)
plt.legend()
plt.xlabel("y in km")
plt.ylabel("(x-x_geo) in km")
plt.title("simulation at beta=0.98")
plt.savefig("plot1.png")


coef_m = np.array(coef_m)



for i in range(2):
    plt.clf()
    plt.scatter(m_list,coef_m.T[i])
    plt.title("C_"+str(i)+"(m)")
    plt.xlabel("mass in natural units")
    plt.ylabel("coefficent size")
    plt.savefig("C_"+str(i)+"_of_m.png")

plt.clf()
for i,m in enumerate(m_list):
    plt.plot(y*l_p/1000,deflection(y,coef_m[i][0], coef_m[i][1])*l_p/1000, label="m="+str(m))
plt.legend()
plt.title("fitt detla_x(y,m)")
plt.xlabel("y in km")
plt.ylabel("(x-x_geo) in km")
plt.savefig("fitt_m.png")


plt.clf()

for i in range(2): #C_1 and C_2
    coeffs_of_coeffs, _ = curve_fit(C_of_m,m_list,coef_m.T[i])
    print("A: ", coeffs_of_coeffs[0])
    print("B: ", coeffs_of_coeffs[1])
    plt.plot(m_list, C_of_m(m_list, coeffs_of_coeffs[0], coeffs_of_coeffs[1]))
    plt.title("fitt of C"+str(i))
    plt.ylabel("coefficent size")
    plt.xlabel("mass in natural units")
    plt.savefig("coeff"+str(i)+".png")
    plt.clf()

y_helper = None


beta_list = [0.5, 0.6, 0.7, 0.8,0.9]

coef_beta = []
for beta in beta_list:
    geodesic = Geodesic(beta, x_0, r_s, r_start)
    x_geo,u_geo = geodesic.get_path(simulation_length=simulation_length, dtau = d_tau)
    corr = Correction(x_0,m,r_s, x_geo, u_geo)
    x,_ = corr.get_path(simulation_length=simulation_length, dtau = d_tau)
    x = x.T
    x_geo = x_geo.T
    y=x[1]*np.sin(x[2])*np.sin(x[3])
    x=x[1]*np.sin(x[2])*np.cos(x[3])
    y_helper = y
    x_geo=x_geo[1]*np.sin(x_geo[2])*np.cos(x_geo[3])
    plt.plot(y*l_p/1000,(x-x_geo)*l_p/1000, label="beta="+str(beta))
    coeffs, _ = curve_fit(deflection,y, (x-x_geo))
    coef_beta.append(coeffs)

plt.legend()
print(m)
plt.xlabel("y in km")
plt.ylabel("(x-x_geo) in km")
plt.title("simulation at m="+str(m))
plt.savefig("plot2.png")

coef_beta = np.array(coef_beta)

for i in range(2):
    plt.clf()
    plt.scatter(beta_list,coef_beta.T[i])
    plt.title("C_"+str(i)+"(beta)")
    plt.xlabel("beta")
    plt.ylabel("coefficent size")
    plt.savefig("C_"+str(i)+"_of_beta.png")

plt.clf()
for beta, coef in zip(beta_list, coef_beta):
    geodesic = Geodesic(beta, x_0, r_s, r_start)
    plt.plot(y_helper*l_p/1000,deflection(y_helper,coef[0], coef[1])*l_p/1000, label="beta="+str(beta))
plt.legend()
plt.title("fitt \delta y(x,\beta)")
plt.xlabel("y in km")
plt.ylabel("(x-x_geo) in km")
plt.savefig("fitt2.png")




