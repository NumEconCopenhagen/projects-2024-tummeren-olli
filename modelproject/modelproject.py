import numpy as np
import matplotlib.pyplot as plt

class SolowModelHumanCapital:
    def __init__(self, alpha=0.3, phi=0.2, g=0.02, omega=0.05, n=0.02, s_K=0.2, s_H=0.1, delta=0.05, L0=100, A0=1, K0=10, H0=5, T=100, endogenous=False):
        self.alpha = alpha
        self.phi = phi
        self.g = g
        self.omega = omega
        self.n = n
        self.s_K = s_K
        self.s_H = s_H
        self.delta = delta
        self.L0 = L0
        self.A0 = A0
        self.K0 = K0
        self.H0 = H0
        self.T = T
        self.endogenous = endogenous
        self.simulate()
        
    def simulate(self):
        # Allocate arrays
        self.L = np.empty(self.T)
        self.A = np.empty(self.T)
        self.K = np.empty(self.T)
        self.H = np.empty(self.T)
        self.Y = np.empty(self.T)
        
        # Initial values
        self.L[0] = self.L0
        self.A[0] = self.A0
        self.K[0] = self.K0
        self.H[0] = self.H0

        for t in range(self.T):
            # Output
            self.Y[t] = self.K[t]**self.alpha * self.H[t]**self.phi * (self.A[t] * self.L[t])**(1-self.alpha-self.phi)

            if t < self.T - 1:
                # Update capital, human capital, labor, and technology
                self.K[t+1] = self.s_K * self.Y[t] + (1 - self.delta) * self.K[t]
                self.H[t+1] = self.s_H * self.Y[t] + (1 - self.delta) * self.H[t]
                self.L[t+1] = (1 + self.n) * self.L[t]
                self.A[t+1] = self.omega * self.H[t] if self.endogenous else (1 + self.g) * self.A[t]

    def plot(self):
        time = np.arange(self.T)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(time, self.Y)
        axs[0, 0].set_title("Output (Y)")
        axs[0, 0].set_xlabel("Time")

        axs[0, 1].plot(time, self.K)
        axs[0, 1].set_title("Physical Capital (K)")
        axs[0, 1].set_xlabel("Time")

        axs[1, 0].plot(time, self.H)
        axs[1, 0].set_title("Human Capital (H)")
        axs[1, 0].set_xlabel("Time")

        axs[1, 1].plot(time, self.A)
        axs[1, 1].set_title("Technology (A)")
        axs[1, 1].set_xlabel("Time")

        plt.tight_layout()
        plt.show()
        
    def steady_state(self):
        L_star = self.L0 * (1 + self.n) ** (self.T - 1)
        H_star = self.H0 * (1 - (1 - self.s_H / (1 - self.delta))**self.T)
        K_star = self.K0 * (1 - (1 - self.s_K / (1 - self.delta))**self.T)
        A_star = self.omega * H_star if self.endogenous else (1 + self.g) * self.A[-1]
        Y_star = K_star**self.alpha * H_star**self.phi * (A_star * L_star)**(1 - self.alpha - self.phi)

        return {"L_star": L_star, "H_star": H_star, "K_star": K_star, "A_star": A_star, "Y_star": Y_star}

    def optimize(self):
        from scipy.optimize import minimize

        def utility(params):
            self.alpha, self.phi = params
            self.simulate()
            return -self.Y[-1]  # Negative of the last output as the utility

        result = minimize(utility, [self.alpha, self.phi], bounds=[(0, 1), (0, 1)])
        self.alpha, self.phi = result.x
        self.simulate()
        return result
