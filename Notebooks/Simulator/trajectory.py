
import numpy as np
from scipy import constants as const


class Trajctory():
    def __init__(self, x_0, r_s):
        self.x_0 = x_0
        self.r_s = r_s

    def norm_of_vector(self, vec, x):
        u_phi = vec[3]
        u_theta = vec[2]
        u_t = vec[0]
        u_r = vec[1]
        r = x[1]
        theta = x[2]
        return (r**3*(r - r_s)*(u_phi**2*np.sin(theta)**2 + u_theta**2) + r**2*u_r**2 - u_t**2*(r - r_s)**2)/(r*(r - r_s))

    


class Geodesic(Trajctory):
    def __init__(self, beta, x_0,r_s, r_start):
        super().__init__(x_0,r_s)
        self.u_0 = u_0 = np.array([0, 0,0,beta/r_start]) 


    def normalize_u(self,u,x):
        """noramlizes u to u^2 = 1"""
        r = x[1]
        theta = x[2]
        u_theta = u[2]
        u_phi = u[3]
        u_r = u[1]
        #u_t = np.sqrt(r*(c**2*r - c**2*r_s + r**3*u_phi**2*np.sin(theta)**2 + r**3*u_theta**2 - r**2*r_s*u_phi**2*np.sin(theta)**2 - r**2*r_s*u_theta**2 + r*u_r**2))/(r - r_s)
        u_t = np.sqrt(r*(r - self.r_s + r**3*u_phi**2*np.sin(theta)**2 + r**3*u_theta**2 - r**2*self.r_s*u_phi**2*np.sin(theta)**2 - r**2*self.r_s*u_theta**2 + r*u_r**2))/(r - self.r_s)
        return np.array([u_t, u_r, u_theta, u_phi])

    def du_dtau(self,u,x):
        r = x[1]
        theta = x[2]
        phi = x[3]

        u_t = u[0]
        u_r = u[1]
        u_theta = u[2]
        u_phi = u[3]

        #print("x : ", x, "\t", "u :", u)

        a_0 = self.r_s*u_r*u_t/(r*(r - self.r_s))
        a_1 = (-2*r**3*(-r + self.r_s)**2*(u_phi**2*np.sin(theta)**2 + u_theta**2) - r**2*self.r_s*u_r**2 + self.r_s*u_t**2*(r - self.r_s)**2)/(2*r**3*(r - self.r_s))
        a_2 = -u_phi**2*np.sin(2*theta)/2 + 2*u_theta*u_r/r
        a_3 = 2*u_phi*(r*u_theta / np.tan(theta) + u_r)/r
        return np.array([-a_0,-a_1,-a_2,-a_3])

    def get_path(self, simulation_length=10000, dtau=1000):
        u_0 = self.normalize_u(self.u_0, self.x_0)
        u = np.ones((simulation_length+1,4))
        x = np.ones((simulation_length+1,4))

        x[0] = self.x_0
        u[0] = self.u_0

        for i in range(simulation_length):
            u[i] = self.normalize_u(u[i], x[i])
            u[i+1] =  self.du_dtau(u[i], x[i])*dtau + u[i]
            x[i+1] =  u[i]*dtau + x[i]

        return x,u
   
class Correction(Trajctory):
    def __init__(self, x_0, m,r_s,x_geo,u_geo):
        super().__init__(x_0,r_s)
        #self.beta = beta
        self.m = m
        self.x_geo = x_geo
        self.u_geo = u_geo
    
    def get_path(self, simulation_length=10000, dtau=1000):
        x = np.ones((simulation_length+1,4))
        v = np.ones((simulation_length,4))
        x[0] = self.x_0

        for i in range(simulation_length):
            v[i] = self.u_geo[i] + self._up_correction(x[i]) 
            x[i+1] =  v[i]*dtau + x[i]
      
        return x,v

    def _up_correction(self,x):
        r = x[1]
        theta = x[2]

        #print((2*r_s*(-r + r_s)**4*np.sin(theta) + r_s*(r - r_s)**2*(r*(r - r_s) + (-r + r_s)**2)*np.sin(theta) - 2*np.sqrt((r - r_s)/r)*(-r + r_s)**4*(2*r*np.sqrt(1 - r_s/r)*np.sin(theta) + np.sin(theta) + 1))/(8*r**3*(-r + r_s)**5*np.sin(theta)) * hbar/m)
        return np.array([0, (r - self.r_s)*np.cos(theta)/(4*self.m*r), (-2*r**2*np.sqrt(1 - self.r_s/r)*np.sin(theta) - 2*r**2*np.sqrt(1 - self.r_s/r) + self.r_s)/(8*self.m*r**4), 0])
        #return hbar/m * np.array([0, -(r - r_s)*(np.sin(theta) + 1)*np.cos(theta)/(4*r*np.sin(theta)**2), (2*r_s*(-r + r_s)**4*np.sin(theta) + r_s*(r - r_s)**2*(r*(r - r_s) + (-r + r_s)**2)*np.sin(theta) - 2*np.sqrt((r - r_s)/r)*(-r + r_s)**4*(2*r*np.sqrt(1 - r_s/r)*np.sin(theta) + np.sin(theta) + 1))/(8*r**3*(-r + r_s)**5*np.sin(theta)), 0])

    
    # def _up_x_correction(self,x):

    #     r = x[1]
    #     theta = x[2]

    #     return np.array([0, 0, 0, hbar*(np.sin(theta) + 1)*np.cos(theta)/(4*m*r**2*np.sin(theta)**4)])

    # def _up_y_correction(self,x):

    #     r = x[1]
    #     theta = x[2]
        
    #     return np.array([0, 0, 0, hbar*(-4*r**2 + 12*r*r_s - 2*r*np.sqrt(1 - r_s/r) - 2*r*np.sqrt(1 - r_s/r)/np.sin(theta) - 7*r_s**2 + 2*r_s*np.sqrt(1 - r_s/r) + 2*r_s*np.sqrt(1 - r_s/r)/np.sin(theta))/(8*m*r**3*(r**2 - 2*r*r_s + r_s**2)*np.sin(theta)**2)])
    
    # def _down_correction(self,x):
    #     r = x[1]
    #     theta = x[2]

    #     pass
    #     #return hbar_m*np.array([0, 0.25*c**2*(-r + r_s)*np.sin(theta)**2*np.cos(theta)/r, - c**2*(r_s*((r - r_s)/r)**(9/2)*(0.25*r**3 - 0.125*c**2*(r - r_s)**3) + 0.25*(r - r_s)**7*(np.sin(theta)**3 + 1))/(r**6*((r - r_s)/r)**(9/2)*(r - r_s)), 0])

