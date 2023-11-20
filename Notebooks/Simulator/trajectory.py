
"""
    ss-metric = mostly plus
    the geodesic equation for 
    dtheta/dtau = 0 and theta = pi/2 reads as follows 
    a^\mu = [r_s* u_r* u_t/(r*(r - r_s)), (-2*r**3*u_\phi**2*(-r + r_s)**2 - r**2*r_s*u_r**2 + r_s*u_t**2*c**2*(r - r_s)**2)/(2*r**3*(r - r_s)), 0, 2*u_\phi*u_r/r]
"""

import numpy as np
from scipy import constants as const

c = const.c

m = 2.14E-37 

M = 5.972E24 
r_s = 2* const.G * M/c**2

hbar_m = const.hbar/m

class Geodesic():
    def __init__(self, u_0, x_0):
        self.u_0 = u_0
        self.x_0 = x_0

        self.corr = Correction(hbar_m=hbar_m)

    def normalize_u(self,u,x):
        """noramlizes u to u^2 = 1"""
        r = x[1]
        u_phi = u[3]
        u_r = u[1]
        #print(r)
        u_t = np.sqrt(r*(c**2*r - c**2*r_s + r**3*u_phi**2 - r**2*r_s*u_phi**2 + r*u_r**2))/(c*(r - r_s))
        return np.array([u_t, u_r, 0, u_phi])

    def acc(self,u,x):
        r = x[1]
        u_t = u[0]
        u_r = u[1]
        u_phi = u[3]
        a_0 = r_s* u_r* u_t/(r*(r - r_s))
        a_1 = (-2*r**3*u_phi**2*(-r + r_s)**2 - r**2*r_s*u_r**2 + r_s*u_t**2*c**2*(r - r_s)**2)/(2*r**3*(r - r_s))
        a_3 = 2*u_phi*u_r/r

        #print(a_3)

        return np.array([a_0,a_1,0,a_3])

    def calc_path(self, simulation_length=10000, dtau=0.0001, spin=0):

        #t,r,theta, phi
        u_0 = self.normalize_u(self.u_0, self.x_0)

        x = np.ones((simulation_length+1,4))
        u = np.ones((simulation_length+1,4))

        x[0] = self.x_0
        u[0] = self.u_0

        if spin==1/2:
            v = np.ones((simulation_length,4))
            for i in range(simulation_length):
                #constrain u^mu u_mu = -1 for all tau
                # => overwrite u_t component

                u[i] = self.normalize_u(u[i], x[i])
                #print(u[i])

                v[i] = u[i] + self.corr.up_correction(x[i])
                
                u[i+1] =  self.acc(u[i], x[i])*dtau + u[i]
                #print(u[i])
                x[i+1] =  v[i]*dtau + x[i]
        if spin==-1/2:
            v = np.ones((simulation_length,4))
            for i in range(simulation_length):
                #constrain u^mu u_mu = -1 for all tau
                # => overwrite u_t component

                u[i] = self.normalize_u(u[i], x[i])
                #print(u[i])

                v[i] = u[i] + self.corr.down_correction(x[i])
                
                u[i+1] =  self.acc(u[i], x[i])*dtau + u[i]
                #print(u[i])
                x[i+1] =  v[i]*dtau + x[i]

        elif spin==0:
            for i in range(simulation_length):

                #constrain u^mu u_mu = -1 for all tau
                # => overwrite u_t component

                u[i] = self.normalize_u(u[i], x[i])
                
                u[i+1] =  self.acc(u[i], x[i])*dtau + u[i]
                #print(u[i])
                x[i+1] =  u[i]*dtau + x[i]


        return x

class Correction():
    def __init__(self, hbar_m):
        self.hbar_m = hbar_m

    def __sigma(self,x):
        r = x[1]
        f = 1 - r_s/r
        return np.array([[0, 0, 0, 0], [0, 0, sqrt(f)/r, 0], [0, -sqrt(f)/r, 0, 0], [0, 0, 0, 0]])

    def up_correction(self,x):
        r = x[1]
        return hbar_m*np.array([0, 0,  (0.25*r**3*r_s - 0.125*r_s*c**2*(r - r_s)**3 + 0.25*(r - r_s)**7*(2)/((r - r_s)/r)**(9/2))/(r**4*(r - r_s)), 0])

    def down_correction(self,x):
        r = x[1]
        return hbar_m*np.array([0, 0, (-0.25*r**3*r_s + 0.125*r_s*c**2*(r - r_s)**3 + 0.25*(r - r_s)**7*(-2)/((r - r_s)/r)**(9/2))/(r**4*(r - r_s)), 0])

