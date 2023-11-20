import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

M = 10

hbar_m = 0.01

R = 21

dt = 0.01


def u(x):
    t = x[0]
    r = x[1]
    theta = x[2]
    phi = x[3]

    u_r = np.sqrt(2*(M*(1/r - 1/R) + 1/2))
    u_t = np.sqrt(r*(2*M + r * u_r**2 - r))/(-2*M + r)
    return np.array([u_t, u_r, 0, 0])


def dv_up(x):
    t = x[0]
    r = x[1]
    theta = x[2]
    phi = x[3]

    dv_r = hbar_m * (-1/16 * (np.cos(theta) - np.cos(3*theta)))
    dv_theta = hbar_m * (1/4 * (-2*M*r**3 - M*(2*M - r)**3 - r**5 * np.sqrt((-2*M + r)/r)*(2*M-r)**2 * (np.sin(theta)*np.sin(theta)**2 + 1)))/(r**4 * (2*M -r))

    return np.array([0,dv_r,dv_theta,0])


def dv_down(x):
    t = x[0]
    r = x[1]
    theta = x[2]
    phi = x[3]

    dv_r = hbar_m *  (1/16 * (np.cos(theta) - np.cos(3*theta)))
    dv_theta = hbar_m * (1/4 * (2*M*r**3 + M*(2*M - r)**3 + r**5 * np.sqrt((-2*M +r )/r) * (2*M-r)**2 *(np.sin(theta)*np.sin(theta)**2 + 1)))/(r**4 * (2*M -r))
    #dv_theta = - hbar_m * (1/4 * (-2*M*r**3 - M*(2*M - r)**3 - r**5 * np.sqrt((-2*M + r)/r)*(2*M-r)**2 * (np.sin(theta)*np.sin(theta)**2 + 1)))/(r**4 * (2*M -r))


    return np.array([0,dv_r,dv_theta,0])

def v_up(x):
    return u(x) + dv_up(x)

def v_down(x):
    return u(x) + dv_down(x)


x_0 = np.array([0,R,0,0])

x_spin_up = x_0
x_list_spin_up = [x_0]

x_spin_down = x_0
x_list_spin_down = [x_0]

x_spin_zero = x_0
x_list_spin_zero = [x_0]


for i in range(10000):
    x_spin_up = x_spin_up + v_up(x_spin_up)*dt
    x_spin_down = x_spin_down + v_down(x_spin_down)*dt
    x_spin_zero = x_spin_zero + u(x_spin_zero)*dt

    x_list_spin_up.append(x_spin_up)
    x_list_spin_down.append(x_spin_down)
    x_list_spin_zero.append(x_spin_zero)


x_list_spin_up = np.array(x_list_spin_up)
x_list_spin_down = np.array(x_list_spin_down)
x_list_spin_zero = np.array(x_list_spin_zero)

#t_list_spin_up = x_list_spin_up.T[0]
#t_list_spin_zero = x_list_spin_zero.T[0]

r_list_spin_up = x_list_spin_up.T[1]
r_list_spin_down = x_list_spin_down.T[1]
r_list_spin_zero = x_list_spin_zero.T[1]

theta_list_spin_up = x_list_spin_up.T[2]
theta_list_spin_down = x_list_spin_down.T[2]
theta_list_spin_zero = x_list_spin_zero.T[2]




fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot( theta_list_spin_up, r_list_spin_up, label="spin-up")
ax.plot( theta_list_spin_down, r_list_spin_down, label="spin-down")
ax.plot( theta_list_spin_zero, r_list_spin_zero, label="spin-zero")
#ax.plot(np.array(theta_list_spin_zero), np.array(x_list_spin_zero), label="spin-zero")
ax.set_rmax(45)
#ax.set_thetamin(0)  
#ax.set_thetamax(np.pi)  
ax.grid(True)


#ax.set_title("A line plot on a polar axis", va='bottom')
#plt.show()
plt.legend()
plt.savefig("trajectory.png")