import numpy as np
from trajectory import Geodesic
import matplotlib.pyplot as plt


r_start = 6371000 #earth radius

x_0 = np.array([0, r_start, np.pi/2, 0])
u_0 = np.array([0, 0,0,1]) 

geo = Geodesic(u_0, x_0)

x_up = geo.calc_path(spin=1/2, simulation_length=3000,dtau = 1E-5).T
x_down = geo.calc_path(spin=-1/2, simulation_length=3000,dtau = 1E-5).T
x_geodesic = geo.calc_path(spin=0, simulation_length=3000, dtau=1E-5).T



plt.plot(x_geodesic[3]*r_start/1000,(np.pi/2-x_geodesic[2])*r_start/1E3, label="spin=0")
plt.plot(x_up[3]*r_start/1000,(np.pi/2-x_up[2])*r_start/1E3, label="spin=1/2 up z-direction")
plt.plot(x_down[3]*r_start/1000,(np.pi/2-x_down[2])*r_start/1E3, label="spin=1/2 down -z-direction")
plt.xlabel("radial distance in km parallel to equator")
plt.ylabel("radial distance in m normal to equator")
plt.legend()
plt.savefig("trajectory_2.png")
