from types import SimpleNamespace
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.stats

class ExchangeEconomyClass:

    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self, x1A, x2A):
        # Utility function for consumer A
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self, x1B, x2B):
        # Utility function for consumer B
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    def demand_A(self, p1):
        # Demand function for consumer A
        x1A = self.par.alpha * ((p1 * self.par.w1A + self.par.w2A) / p1)
        x2A = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.w2A)
        return x1A, x2A

    def demand_B(self, p1):
        # Demand function for consumer B
        x1B = self.par.beta * ((p1 * (1 - self.par.w1A) + (1 - self.par.w2A)) / p1)
        x2B = (1 - self.par.beta) * (p1 * (1 - self.par.w1A) + (1 - self.par.w2A))
        return x1B, x2B

    def check_market_clearing(self, p1):
        par = self.par

        # Calculate demand for both consumers
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)

        # Calculate market clearing conditions
        eps1 = x1A - par.w1A + x1B - (1 - par.w1A)
        eps2 = x2A - par.w2A + x2B - (1 - par.w2A)

        return eps1, eps2


    
    def calculate_possible_allocations(self, N=75):
        # Generate possible allocations
        x1A = np.linspace(0, 1, N+1)
        x2A = np.linspace(0, 1, N+1)
        x1possible = []
        x2possible = []
        for x1 in x1A:
            for x2 in x2A:
                if self.utility_A(x1, x2) >= self.utility_A(self.par.w1A, self.par.w2A) and self.utility_B(1-x1, 1-x2) >= self.utility_B(1-self.par.w1A, 1-self.par.w2A):
                    x1possible.append(x1)
                    x2possible.append(x2)
        return x1possible, x2possible

    def plot_edgeworth_box(self, x1possible, x2possible):
        par = self.par
        # a. We start off by making the total endowment
        w1bar = 1.0
        w2bar = 1.0
        # b. We then setup the figure details
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()
        # A
        ax_A.scatter(par.w1A, par.w2A, marker='s', color='black', label='endowment')
        ax_A.scatter(x1possible, x2possible, marker='o', alpha=0.2, color='cyan', label='possible allocations')
        # At last, we add limits to the plot
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')
        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])
        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))
        plt.show()

    def calculate_market_clearing_errors(self, p1_values):
        # We calculate the market clearing errors for a range of prices
        errors = [self.check_market_clearing(p1) for p1 in p1_values]
        error_term_1 = [error[0] for error in errors]
        error_term_2 = [error[1] for error in errors]
        return error_term_1, error_term_2

    def plot_market_clearing_errors(self, p1_values, error_term_1, error_term_2):
        # We plot market clearing errors
        plt.plot(p1_values, error_term_1, label='$\epsilon_1(p,\omega)$')
        plt.plot(p1_values, error_term_2, label='$\epsilon_2(p,\omega)$')
        # We then add labels and a legend to the plot
        plt.xlabel('p1')
        plt.ylabel('Error with market clearing')
        plt.legend()
        plt.show()

    def optimize_price(self, objective_function, initial_guess=1.0, bounds=[(0.5, 2.5)]):
        # We minimize the objective function to find the optimal price
        result = minimize(objective_function, initial_guess, bounds=bounds)
        return result.x[0]

    def find_market_clearing_price(self):
        # We define the objective function for market clearing
        def objective_function(p1):
            eps1, eps2 = self.check_market_clearing(p1)
            return eps1 ** 2 + eps2 ** 2

        # Then find the price that minimizes the market clearing error
        return self.optimize_price(objective_function, initial_guess=1.0, bounds=[(0.5, 2.5)])
    
    def find_optimal_allocation(self):
        # We define the objective function to maximize A's utility
        def objective_function(p1):
            x1B, x2B = self.demand_B(p1)
            x1A = 1 - x1B
            x2A = 1 - x2B
            return -self.utility_A(x1A, x2A)  # Negative because we want to maximize

        # We find the price that maximizes A's utility
        optimal_p1 = self.optimize_price(objective_function, initial_guess=1.0, bounds=[(0.0001, None)])

        # Then we calculate the corresponding allocation
        x1B, x2B = self.demand_B(optimal_p1)
        x1A = 1 - x1B
        x2A = 1 - x2B

        return optimal_p1, x1A, x2A

    def find_market_clearing_price(self):
        # We define the objective function for market clearing
        def objective_function(p1):
            eps1, eps2 = self.check_market_clearing(p1)
            return eps1 ** 2 + eps2 ** 2

        # Then find the price that minimizes the market clearing error
        return self.optimize_price(objective_function, initial_guess=1.0, bounds=[(0.5, 2.5)])
    
    def find_best_allocation_with_p1_set(self, p1_values):
        # We start off by finding the best allocation for given price sets
        util_A_best = -np.inf
        best_allocation_p1 = None
        for t in p1_values:
            # We calculate the demand for B
            x1B, x2B = self.demand_B(t)
            # We then ensure that A ends up with a positive amount of both goods
            if 1 - x1B > 0 and 1 - x2B > 0:
                # So we calculate the utility of A
                util_A = self.utility_A(1 - x1B, 1 - x2B)
                # We then update util_A_best if util_A is larger
                if util_A > util_A_best:
                    util_A_best = util_A
                    best_allocation_p1 = t
        # At last we calculate the corresponding allocation for A
        x1B_best, x2B_best = self.demand_B(best_allocation_p1)
        x1A_best = 1 - x1B_best
        x2A_best = 1 - x2B_best
        return best_allocation_p1, util_A_best, x1A_best, x2A_best
    
    def find_best_allocation_in_C(self, x1possible, x2possible):
        # We want to find the best allocation in set C. We start off by setting the starting point to -infinity 
        util_A_better = -np.inf
        x_1_op = None
        x_2_op = None
        #We use our results from question 1 to make a list, C that contains all possible combinations of x1 and x2.
        C = zip(x1possible, x2possible)
        #We then loop over the list C, and find the combination of x1 and x2 that gives the highest utility for A, and also gives a utility for B that is higher than the utility of B at the initial endownment. 
        for x_1, x_2 in C:
            util_A_start = self.utility_A(x_1, x_2)
            if self.utility_A(x_1, x_2) >= util_A_better and self.utility_B(1 - x_1, 1 - x_2) >= self.utility_B(1 - 0.8, 1 - 0.3):
                util_A_better = util_A_start
                x_1_op = x_1
                x_2_op = x_2
        return util_A_better, x_1_op, x_2_op, self.utility_B(1 - x_1_op, 1 - x_2_op)

    def find_best_allocation_no_restriction(self):
        # We start off by defining the objective function to maximize A's utility
        def objective_function(x):
            x1A, x2A = x
            return -self.utility_A(x1A, x2A)  # Negative because we want to maximize

        # We define the constraint for B's utility
        def constraint(x):
            x1A, x2A = x
            x1B = 1 - x1A
            x2B = 1 - x2A
            return self.utility_B(x1B, x2B) - self.utility_B(1 - self.par.w1A, 1 - self.par.w2A)

        # We make our initial guess for x1A and x2A (We use the answer from 5a as initial guess because there might be a non-global minima) 
        initial_guess = [0.602, 0.823]

        # We add the constraints definition
        constraints = {
            'type': 'ineq',
            'fun': constraint
        }

        # We then add bounds for x1A and x2A
        bounds = [(0, 1), (0, 1)]

        # We use the earlier defined objective_function to find the optimal allocation. 
        result = minimize(objective_function, initial_guess, bounds=bounds, constraints=constraints)

        # Optimal values
        x_1_opti = result.x[0]
        x_2_opti = result.x[1]

        return x_1_opti, x_2_opti, self.utility_A(x_1_opti, x_2_opti), self.utility_B(1-x_1_opti,1- x_2_opti)

    
    
    def find_social_planner_allocation(self):
        # we first define the objective function to maximize the sum of utilities
        def objective_function(x):
            x1A, x2A = x
            x1B = 1 - x1A
            x2B = 1 - x2A
            return -(self.utility_A(x1A, x2A) + self.utility_B(x1B, x2B))  # Negative because we want to maximize

        # We make our initial guess for x1A and x2A
        initial_guess = [0.5, 0.5]

        # We then set the bounds for x1A and x2A
        bounds = [(0, 1), (0, 1)]

        # Now we perform the optimization 
        result = minimize(objective_function, initial_guess, bounds=bounds)

        # We check the optimal values
        x1A_opti_social = result.x[0]
        x2A_opti_social = result.x[1]
        x1B_opti_social = 1 - x1A_opti_social
        x2B_opti_social = 1 - x2A_opti_social
        util_A_opt = self.utility_A(x1A_opti_social, x2A_opti_social)
        util_B_opt = self.utility_B(x1B_opti_social, x2B_opti_social)
        util_A_social = util_A_opt + util_B_opt

        return util_A_social, x1A_opti_social, x2A_opti_social, util_A_opt, util_B_opt

    def plot_all_allocations(self, x1possible, x2possible,best_allocation_4a, best_allocation_4b, best_allocation_5a, best_allocation_5b, best_allocation_6a):
        par = self.par
        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0
        # b. We set up the figure 
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()
        # We plot the possible allocations and add the best allocations from the different questions
        ax_A.scatter(par.w1A, par.w2A, marker='s', color='black', label='endowment')
        ax_A.scatter(x1possible, x2possible, marker='o', alpha=0.1, color='cyan', label='possible allocations')
        ax_A.scatter(best_allocation_4a[1], best_allocation_4a[2], marker='s', color='green', label='4a. best allocation')
        ax_A.scatter(best_allocation_4b[1], best_allocation_4b[2], marker='s', color='grey',alpha=0.7, label='4b. best allocation')
        ax_A.scatter(best_allocation_5a[1], best_allocation_5a[2], marker='s', color='darkviolet', label='5a. best allocation')
        ax_A.scatter(best_allocation_5b[1], best_allocation_5b[2], marker='s', color='gold', label='5b. best allocation')
        ax_A.scatter(best_allocation_6a[1], best_allocation_6a[2], marker='s', color='lime', label='6a. social planner')
        # We add limits to the plot 
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')
        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])
        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))
        plt.show()

    def plot_random_allocations(self):
        # We generate and plot a random set of allocations with seed 2000 for replication
        np.random.seed(2000)
        omega_1 = np.random.uniform(0, 1, 50)
        omega_2 = np.random.uniform(0, 1, 50)
        plt.scatter(omega_1, omega_2, color='blue', marker='o', alpha=0.8)
        plt.xlabel('$\omega_1$')
        plt.ylabel('$\omega_2$')
        plt.title('Random set of allocations from $\omega$')
        plt.grid(True)
        plt.show()
    
    def generate_random_endowments(self, num_elements=50):
        np.random.seed(2000)
        omega_1 = np.random.uniform(0, 1, num_elements)
        omega_2 = np.random.uniform(0, 1, num_elements)
        return list(zip(omega_1, omega_2))

    def find_market_equilibrium_allocation(self, w1A, w2A):
        self.par.w1A = w1A
        self.par.w2A = w2A

        # We find the market clearing price
        market_clearing_p1 = self.find_market_clearing_price()

        # Then we calculate the corresponding allocation
        x1A, x2A = self.demand_A(market_clearing_p1)
        x1B, x2B = self.demand_B(market_clearing_p1)

        return x1A, x2A, x1B, x2B, market_clearing_p1

    def plot_random_endowments(self, W):
        #We set up the plot
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        for w1A, w2A in W:
            x1A, x2A, x1B, x2B, p1 = self.find_market_equilibrium_allocation(w1A, w2A)
            ax_A.scatter(x1A, x2A, marker='o', color='blue', alpha=0.5)
            ax_B.scatter(x1B, x2B, marker='o', color='red', alpha=0.5)

        # We add the limits to the plot
        ax_A.plot([0, 1], [0, 0], lw=2, color='black')
        ax_A.plot([0, 1], [1, 1], lw=2, color='black')
        ax_A.plot([0, 0], [0, 1], lw=2, color='black')
        ax_A.plot([1, 1], [0, 1], lw=2, color='black')
        ax_A.set_xlim([-0.1, 1.1])
        ax_A.set_ylim([-0.1, 1.1])
        ax_B.set_xlim([1.1, -0.1])
        ax_B.set_ylim([1.1, -0.1])
        plt.title('Market Equilibrium Allocations for Random Endowments')
        plt.show()
