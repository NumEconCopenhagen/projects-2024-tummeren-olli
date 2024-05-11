
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
class SolowModelHumanCapital:

    def __init__(self, alpha=0.3, phi=0.2, g=0.02, n=0.02, s_K=0.2, s_H=0.1, delta=0.05, L0=1, A0=1, K0=1, H0=1, T=300, Increase_saving=False):
        self.params = {
            'alpha': alpha, 'phi': phi, 'g': g, 'n': n,
            's_K': s_K, 's_H': s_H, 'delta': delta, 'L0': L0, 'A0': A0,
            'K0': K0, 'H0': H0, 'T': T, 'increase_saving': Increase_saving
        }
        self.simulate()
    def simulate(self):
        # Unpack parameters
        p = self.params
        # Allocate arrays
        self.L = np.empty(p['T'])
        self.A = np.empty(p['T'])
        self.K = np.empty(p['T'])
        self.H = np.empty(p['T'])
        self.Y = np.empty(p['T'])
        self.K_tilde = np.empty(p['T'])
        self.H_tilde = np.empty(p['T'])
        self.Y_tilde = np.empty(p['T'])
        # Initial values
        self.L[0] = 1
        self.A[0] = 1 
        self.K[0] = 1
        self.H[0] = 1 
        self.K_tilde[0] = p['K0'] / (p['A0'] * p['L0'])
        self.H_tilde[0] = p['H0'] / (p['A0'] * p['L0'])
        for t in range(p['T']-1):
            self.Y_tilde[t] = self.K_tilde[t]**p['alpha'] * self.H_tilde[t]**p['phi']
            self.A[t+1] = (1 + p['g']) * self.A[t]
            self.L[t+1] = (1 + p['n']) * self.L[t]
            growth_denominator = (1 + p['n']) * (1 + p['g'])
            if p['increase_saving'] and t > 0:
                p['s_H'] += 0.00015 * self.Y_tilde[t-1]
            self.K_tilde[t + 1] = (p['s_K'] * self.Y_tilde[t] + (1 - p['delta']) * self.K_tilde[t]) / growth_denominator
            self.H_tilde[t + 1] = (p['s_H'] * self.Y_tilde[t] + (1 - p['delta']) * self.H_tilde[t]) / growth_denominator
            
            self.K[t+1] = self.K_tilde[t+1] * self.A[t+1] * self.L[t+1]
            self.H[t+1] = self.H_tilde[t+1] * self.A[t+1] * self.L[t+1]
        self.Y[-1] = self.K[-1]**p['alpha'] * self.H[-1]**p['phi'] * (self.A[-1] * self.L[-1])**(1-p['alpha']-p['phi'])
    def update_params(self, **kwargs):
        self.params.update(kwargs)
        self.simulate()
    def plot_tilde_variables(self):
        time = np.arange(self.params['T'])
        plt.figure(figsize=(12, 8))
        plt.plot(time, self.K_tilde, label='Capital per Effective Worker ($\\tilde{k}$)')
        plt.plot(time, self.H_tilde, label='Human Capital per Effective Worker ($\\tilde{h}$)')
        plt.title('Evolution of Capital Stocks per Effective Worker')
        plt.xlabel('Time')
        plt.ylabel('Capital Stocks per Effective Worker')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def steady_state(self):
        # Define the system of equations
        def equations(vars):
            k_tilde, h_tilde = vars
            p = self.params
            eq1 = p['s_K'] * k_tilde**p['alpha'] * h_tilde**p['phi'] - (p['n'] + p['g'] + p['delta'] + p['n']*p['g']) * k_tilde
            eq2 = p['s_H'] * k_tilde**p['alpha'] * h_tilde**p['phi'] - (p['n'] + p['g'] + p['delta'] + p['n']*p['g']) * h_tilde
            return [eq1, eq2]

        # Initial guess
        k_tilde_guess = 1
        h_tilde_guess = 1

        # Solve for steady state
        k_tilde_ss, h_tilde_ss = fsolve(equations, (k_tilde_guess, h_tilde_guess))

        return k_tilde_ss, h_tilde_ss
