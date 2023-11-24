import numpy as np
from trajectory import Geodesic
import matplotlib.pyplot as plt
from scipy import constants as const

#c = const.c

l = 1.9732705E-7

r_start = 6371000 /l #earth radius in natural units


x_0 = np.array([0, r_start, np.pi/2, 0])


beta = 0.98

u_0 = np.array([0, 0,0,beta/r_start]) 



geo = Geodesic(u_0, x_0)

(x_up,u,correction) = geo.calc_path(spin=1/2, simulation_length=500000,dtau = 10131712) 
#x_down = geo.calc_path(spin=-1/2, simulation_length=500,dtau = 1E-8).T
(x_geodesic,_,_) = geo.calc_path(spin=0, simulation_length=500000, dtau=10131712)

np.savetxt("u.csv", u, delimiter=",")
np.savetxt("correction.csv", correction, delimiter=",")
np.savetxt("position.csv", x_up, delimiter=",")

x_up = x_up.T
x_geodesic = x_geodesic.T

plt.scatter(x_geodesic[3][:-2]*r_start*l/1000, (np.pi/2-x_geodesic[2][:-2])*r_start*l, label="gedesic")
plt.scatter(x_geodesic[3][:-2]*r_start*l/1000, (correction.T[2][0]-correction.T[2][:-2])*r_start*l * 1e12*2, label="correction")
plt.scatter(x_up[3][:-2]*r_start*l/1000,(np.pi/2-x_up[2][:-2])*r_start*l, label="spin=1/2")
plt.scatter(x_geodesic[3][:-2]*r_start*l/1000,(np.pi/2-x_up[2][:-2])*r_start*l, label="spin=1/2")
#plt.scatter(x_up[3]*r_start*l/1000,(np.pi/2-x_up[2])*r_start*l, label="spin=1/2 up z-direction")
#plt.plot(x_down[3]*r_start/1000,(np.pi/2-x_down[2])*r_start, label="spin=1/2 down -z-direction")

#diff
#plt.plot(x_geodesic[3]*r_start*l,(x_up[2]-x_geodesic[2]), label="difference")


plt.xlabel("distance in km parallel to equator")
plt.ylabel("distance in m normal to equator")
plt.legend()
plt.savefig("trajectory_2.png")
