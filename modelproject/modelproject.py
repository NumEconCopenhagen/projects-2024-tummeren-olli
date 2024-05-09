
import numpy as np
import matplotlib.pyplot as plt
class SolowModelHumanCapital:
    def __init__(self, alpha=0.3, phi=0.2, g=0.02, omega=0.05, n=0.02, s_K=0.2, s_H=0.1, delta=0.05, L0=100, A0=1, K0=10, H0=5, T=200, endogenous=False):
        self.params = {
            'alpha': alpha, 'phi': phi, 'g': g, 'omega': omega, 'n': n,
            's_K': s_K, 's_H': s_H, 'delta': delta, 'L0': L0, 'A0': A0,
            'K0': K0, 'H0': H0, 'T': T, 'endogenous': endogenous
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
        self.L[0], self.A[0], self.K[0], self.H[0] = p['L0'], p['A0'], p['K0'], p['H0']
        self.K_tilde[0] = p['K0'] / (p['A0'] * p['L0'])
        self.H_tilde[0] = p['H0'] / (p['A0'] * p['L0'])
        for t in range(p['T']-1):
            self.Y_tilde[t] = self.K_tilde[t]**p['alpha'] * self.H_tilde[t]**p['phi']
            if p['endogenous']:
                self.A[t+1] = p['omega'] * self.H[t]
            else:
                self.A[t+1] = (1 + p['g']) * self.A[t]
            
            self.L[t+1] = (1 + p['n']) * self.L[t]
            growth_denominator = (1 + p['n']) * (1 + p['g']) if not p['endogenous'] else (1 + p['n'])
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
        plt.plot(time, self.K_tilde, label='Capital per Effective Worker ($\\tilde{K}$)')
        plt.plot(time, self.H_tilde, label='Human Capital per Effective Worker ($\\tilde{H}$)')
        plt.title('Evolution of Capital Stocks per Effective Worker')
        plt.xlabel('Time')
        plt.ylabel('Capital Stocks per Effective Worker')
        plt.legend()
        plt.grid(True)
        plt.show()
