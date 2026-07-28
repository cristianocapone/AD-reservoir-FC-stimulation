"""
© 2025 This work is licensed under a CC-BY-NC-SA license.
Title:
**Authors:** Cristiano Capone
"""

import numpy as np


class RESERVOIRE_SIMPLE:

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];

        self.dt = par['dt']#1. / self.T;
        self.tau_m = np.linspace(  par['tau_m_f'],  par['tau_m_s'] ,self.N)

        self.J = np.random.normal (0., 1./np.sqrt(self.N), size = (self.N, self.N));#np.zeros ((self.N, self.N));
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))

        self.Jout = np.random.normal (0.0, 0., size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.h_Jout = np.zeros((self.O,))
        self.y = np.zeros((self.O,))
        # Membrane potential
        self.X = np.zeros (self.N)

        # Here we save the params dictionary
        self.par = par

    def step_rate (self, inp, sigma_dyn=0):

        self.X_noisy   = np.copy(self.X) + np.random.normal (0., sigma_dyn, size = (self.N,) )
        self.X   = self.X   * np.exp(-self.dt/self.tau_m) + (1.-np.exp(-self.dt/self.tau_m))  * (self.J @ self.X_noisy   + self.Jin @ inp ) 
        self.y = self.Jout@ self.X + self.h_Jout#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        return self.X

    def reset (self, init = None):
        self.X  =  np.zeros ((self.N,))#*= 0.;
        self.y = np.zeros((self.O,))
