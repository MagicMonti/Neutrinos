
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
hbar = 1
l_p = 1.9732705E-7
m_p = 1.7826627E-36 


#r_s in natrual units
r_s = r_s/l_p
m = m / m_p

m = m * 1E-8


print("r_s", r_s)
print("inverse mass", 1/m)


class Geodesic():
    def __init__(self, u_0, x_0):
        self.u_0 = u_0
        self.x_0 = x_0

        self.corr = Correction()

    def normalize_u(self,u,x):
        """noramlizes u to u^2 = 1"""
        r = x[1]
        theta = x[2]
        u_theta = u[2]
        u_phi = u[3]
        u_r = u[1]
        #u_t = np.sqrt(r*(c**2*r - c**2*r_s + r**3*u_phi**2*np.sin(theta)**2 + r**3*u_theta**2 - r**2*r_s*u_phi**2*np.sin(theta)**2 - r**2*r_s*u_theta**2 + r*u_r**2))/(r - r_s)
        u_t = np.sqrt(r*(r - r_s + r**3*u_phi**2*np.sin(theta)**2 + r**3*u_theta**2 - r**2*r_s*u_phi**2*np.sin(theta)**2 - r**2*r_s*u_theta**2 + r*u_r**2))/(r - r_s)
        return np.array([u_t, u_r, u_theta, u_phi])

    # def acc(self,u,x):
    #     r = x[1]
    #     theta = x[2]
    #     u_theta = u[2]
    #     u_phi = u[3]
    #     u_r = u[1]
    #     return hbar_m*np.array([0, -r_s*u_theta*np.sqrt(1 - r_s)/r**2, r_s*u_r*np.sqrt(1 - r_s)/(r**3*(r - r_s)), 0])

    def du_dtau(self,u,x):
        r = x[1]
        theta = x[2]
        phi = x[3]

        u_t = u[0]
        u_r = u[1]
        u_theta = u[2]
        u_phi = u[3]

        #print("x : ", x, "\t", "u :", u)

        a_0 = r_s*u_r*u_t/(r*(r - r_s))
        a_1 = (-2*r**3*(-r + r_s)**2*(u_phi**2*np.sin(theta)**2 + u_theta**2) - r**2*r_s*u_r**2 + r_s*u_t**2*(r - r_s)**2)/(2*r**3*(r - r_s))
        a_2 = -u_phi**2*np.sin(2*theta)/2 + 2*u_theta*u_r/r
        a_3 = 2*u_phi*(r*u_theta / np.tan(theta) + u_r)/r
        return np.array([-a_0,-a_1,-a_2,-a_3])

    def norm_of_vector(self, vec, x):

        u_phi = vec[3]
        u_theta = vec[2]
        u_t = vec[0]
        u_r = vec[1]
        r = x[1]
        theta = x[2]

        return (r**3*(r - r_s)*(u_phi**2*np.sin(theta)**2 + u_theta**2) + r**2*u_r**2 - u_t**2*(r - r_s)**2)/(r*(r - r_s))

    def calc_path(self, simulation_length=10000, dtau=0.0001, spin=0):

        #t,r,theta, phi
        u_0 = self.normalize_u(self.u_0, self.x_0)

        x = np.ones((simulation_length+1,4))
        x_geo = np.ones((simulation_length+1,4))
        u = np.ones((simulation_length+1,4))

        x[0] = self.x_0
        u[0] = self.u_0

        x_geo[0] = self.x_0

        correction = np.ones((simulation_length+1,4))

        if spin==1/2:
            v = np.ones((simulation_length,4))
            for i in range(simulation_length):
                #constrain u^mu u_mu = -1 for all tau
                # => overwrite u_t component

                u[i] = self.normalize_u(u[i], x_geo[i])
                #print(u[i])

                #print(self.normalize_u(u[i], x[i]))
                
                #print(self.norm_of_vector(u[i], x[i])/c**2,self.norm_of_vector(self.corr.up_correction(x[i]),x[i])/c**2)
                v[i] = u[i] + self.corr.up_correction(x[i]) 

                correction[i] = self.corr.up_correction(x[i]) 

                
                u[i+1] =  self.du_dtau(u[i], x_geo[i])*dtau + u[i]
                #print(u[i])
                x[i+1] =  v[i]*dtau + x[i]
                x_geo[i+1] = u[i]*dtau + x_geo[i]

                #print(self.acc(u[i], x[i]))

                #print(self.norm_of_vector(u[i], x[i]), self.norm_of_vector(v[i], x[i]))
        if spin==-1/2:
            v = np.ones((simulation_length+1,4))
            v[0] = self.normalize_u(u[0], x[0]) #+ self.corr.down_correction(x[0])
            #print(self.normalize_u(u[0], x[0]), self.corr.down_correction(x[0]))
            for i in range(simulation_length):

               pass

        elif spin==0:
           
            for i in range(simulation_length):

                u[i] = self.normalize_u(u[i], x[i])

                
                u[i+1] =  self.du_dtau(u[i], x[i])*dtau + u[i]

           
                x[i+1] =  u[i]*dtau + x[i]
                

        return (x,u,correction)

class Correction():
    def __init__(self):
        pass

    # def __sigma(self,x):
    #     r = x[1]
    #     f = 1 - r_s/r
    #     return np.array([[0, 0, 0, 0], [0, 0, sqrt(f)/r, 0], [0, -sqrt(f)/r, 0, 0], [0, 0, 0, 0]])

    def up_correction(self,x):
        r = x[1]
        theta = x[2]

        #print((2*r_s*(-r + r_s)**4*np.sin(theta) + r_s*(r - r_s)**2*(r*(r - r_s) + (-r + r_s)**2)*np.sin(theta) - 2*np.sqrt((r - r_s)/r)*(-r + r_s)**4*(2*r*np.sqrt(1 - r_s/r)*np.sin(theta) + np.sin(theta) + 1))/(8*r**3*(-r + r_s)**5*np.sin(theta)) * hbar/m)
        return np.array([0, hbar*(r - r_s)*np.cos(theta)/(4*m*r), hbar*(-2*r**2*np.sqrt(1 - r_s/r)*np.sin(theta) - 2*r**2*np.sqrt(1 - r_s/r) + r_s)/(8*m*r**4), 0])
        #return hbar/m * np.array([0, -(r - r_s)*(np.sin(theta) + 1)*np.cos(theta)/(4*r*np.sin(theta)**2), (2*r_s*(-r + r_s)**4*np.sin(theta) + r_s*(r - r_s)**2*(r*(r - r_s) + (-r + r_s)**2)*np.sin(theta) - 2*np.sqrt((r - r_s)/r)*(-r + r_s)**4*(2*r*np.sqrt(1 - r_s/r)*np.sin(theta) + np.sin(theta) + 1))/(8*r**3*(-r + r_s)**5*np.sin(theta)), 0])

    
    def up_x_correction(self,x):

        r = x[1]
        theta = x[2]

        return np.array([0, 0, 0, hbar*(np.sin(theta) + 1)*np.cos(theta)/(4*m*r**2*np.sin(theta)**4)])

    def up_y_correction(self,x):

        r = x[1]
        theta = x[2]
        
        return np.array([0, 0, 0, hbar*(-4*r**2 + 12*r*r_s - 2*r*np.sqrt(1 - r_s/r) - 2*r*np.sqrt(1 - r_s/r)/np.sin(theta) - 7*r_s**2 + 2*r_s*np.sqrt(1 - r_s/r) + 2*r_s*np.sqrt(1 - r_s/r)/np.sin(theta))/(8*m*r**3*(r**2 - 2*r*r_s + r_s**2)*np.sin(theta)**2)])
    
    def down_correction(self,x):
        r = x[1]
        theta = x[2]

        pass
        #return hbar_m*np.array([0, 0.25*c**2*(-r + r_s)*np.sin(theta)**2*np.cos(theta)/r, - c**2*(r_s*((r - r_s)/r)**(9/2)*(0.25*r**3 - 0.125*c**2*(r - r_s)**3) + 0.25*(r - r_s)**7*(np.sin(theta)**3 + 1))/(r**6*((r - r_s)/r)**(9/2)*(r - r_s)), 0])

