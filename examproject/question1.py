import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy.optimize import minimize_scalar, root, minimize

# We define the parameters
par = SimpleNamespace()
par.A = 1.0
par.gamma = 0.5
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0
par.tau = 0.0
par.T = 0.0
w = 1.0  # numeraire
par.kappa = 0.1

# We define functions for optimal firm behavior
def optimal_labor(p_j, w, A, gamma):
    return (p_j * A * gamma / w) ** (1 / (1 - gamma))

def optimal_output(l_j, A, gamma):
    return A * (l_j ** gamma)

def optimal_profit(w, p_j, A, gamma):
    return (1 - gamma) / gamma * w * (p_j * A * gamma / w) ** (1 / (1 - gamma))

# We define functions for consumer behavior
def c1_optimal(l, w, p1, p2, alpha, T, pi1, pi2):
    return alpha * (w * l + T + pi1 + pi2) / p1

def c2_optimal(l, w, p1, p2, alpha, T, pi1, pi2, tau):
    return (1 - alpha) * (w * l + T + pi1 + pi2) / (p2 + tau)

def utility_maximization(l, w, p1, p2, alpha, nu, epsilon, T, pi1, pi2, tau):
    c1 = c1_optimal(l, w, p1, p2, alpha, T, pi1, pi2)
    c2 = c2_optimal(l, w, p1, p2, alpha, T, pi1, pi2, tau)
    utility = np.log(c1 ** alpha * c2 ** (1 - alpha)) - nu * (l ** (1 + epsilon)) / (1 + epsilon)
    return -utility  # We minimize the negative utility function to maximize utility

# We now define the market clearing conditions and checks if any price in the grids are satisfying the conditions
def check_market_clearing(par, w, p1_grid, p2_grid):
    results = []
    equilibrium_found = False

    for p1 in p1_grid:
        for p2 in p2_grid:
            # We set up functions to calculate firm behavior
            l1_star = optimal_labor(p1, w, par.A, par.gamma)
            l2_star = optimal_labor(p2, w, par.A, par.gamma)
            y1_star = optimal_output(l1_star, par.A, par.gamma)
            y2_star = optimal_output(l2_star, par.A, par.gamma)
            pi1_star = optimal_profit(w, p1, par.A, par.gamma)
            pi2_star = optimal_profit(w, p2, par.A, par.gamma)
            
            # We then maximize consumers' utility, which is the optimal consumer behavior
            res = minimize_scalar(
                utility_maximization, 
                bounds=(0, 100), 
                args=(w, p1, p2, par.alpha, par.nu, par.epsilon, par.T, pi1_star, pi2_star, par.tau), 
                method='bounded'
            )
            l_star = res.x
            c1_star = c1_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star)
            c2_star = c2_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star, par.tau)
            
            # Here we define the market clearing conditions
            labor_clearing = np.isclose(l_star, l1_star + l2_star)
            good1_clearing = np.isclose(c1_star, y1_star)
            good2_clearing = np.isclose(c2_star, y2_star)
            
            # If all conditions are satisfied, we set equilibrium_found to True
            if labor_clearing and good1_clearing and good2_clearing:
                equilibrium_found = True

            results.append({
                'p1': p1, 
                'p2': p2, 
                'labor_clearing': labor_clearing, 
                'good1_clearing': good1_clearing, 
                'good2_clearing': good2_clearing
            })
    # We then create a DataFrame with the results to show if the different conditions are satisfied or not
    results_df = pd.DataFrame(results)
    # We add a "If" statement that prints if equilibrium is found or not
    if not equilibrium_found:
        print("There is no market clearing equilibrium in the price grids.")
    else:
        print("Market clearing equilibrium found in the price grids.")

    return results_df

# We define the function that finds the equilibrium prices similiar to the function above, without using the price grids
def market_clearing_conditions(prices, par, w):
    p1, p2 = prices

    # Same as before
    l1_star = optimal_labor(p1, w, par.A, par.gamma)
    l2_star = optimal_labor(p2, w, par.A, par.gamma)
    y1_star = optimal_output(l1_star, par.A, par.gamma)
    y2_star = optimal_output(l2_star, par.A, par.gamma)
    pi1_star = optimal_profit(w, p1, par.A, par.gamma)
    pi2_star = optimal_profit(w, p2, par.A, par.gamma)
    
    # Same as before
    res = minimize_scalar(
        utility_maximization, 
        bounds=(0, 100), 
        args=(w, p1, p2, par.alpha, par.nu, par.epsilon, par.T, pi1_star, pi2_star, par.tau), 
        method='bounded'
    )
    l_star = res.x
    c1_star = c1_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star)
    c2_star = c2_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star, par.tau)
    
    # Market clearing conditions but only clearing 2 markets because of walras' law
    labor_clearing = l_star - (l1_star + l2_star)
    good1_clearing = c1_star - y1_star
    
    return np.array([labor_clearing, good1_clearing])

# We define the function that finds the equilibrium prices using root finder on the market clearing conditions above
def find_equilibrium_prices(par, w, initial_guess=[1.0, 1.0]):
    
    # This finds the equilibrium prices
    solution = root(market_clearing_conditions, initial_guess, args=(par, w))

    # The price values gets extracted from the solution
    equilibrium_prices = solution.x
    p1_star, p2_star = equilibrium_prices

    # We print the equilibrium prices
    equilibrium_prices_df_1 = print(f"The equilibrium prices are p1: {p1_star:.4f} and p2: {p2_star:.4f}")

    return equilibrium_prices_df_1

# We define the function that finds the optimal tax and T where all the earlier definitions are used locally in this function. We do this because we had problems running the earlier functions in this question, which might be caused by global variables
def find_optimal_tax_and_transfer():
    # Define the parameters
    par = SimpleNamespace()
    par.A = 1.0
    par.gamma = 0.5
    par.alpha = 0.3
    par.nu = 1.0
    par.epsilon = 2.0
    par.kappa = 0.1
    par.tau = 0.0
    par.T = 0.0
    w = 1.0  # numeraire

    def optimal_labor(p_j, w, A, gamma):
        return (p_j * A * gamma / w) ** (1 / (1 - gamma))

    def optimal_output(l_j, A, gamma):
        return A * (l_j ** gamma)

    def optimal_profit(w, p_j, A, gamma):
        return (1 - gamma) / gamma * w * (p_j * A * gamma / w) ** (1 / (1 - gamma))

    def c1_optimal(l, w, p1, p2, alpha, T, pi1, pi2):
        return alpha * (w * l + T + pi1 + pi2) / p1

    def c2_optimal(l, w, p1, p2, alpha, T, pi1, pi2, tau):
        return (1 - alpha) * (w * l + T + pi1 + pi2) / (p2 + tau)

    def utility_maximization(l, w, p1, p2, alpha, nu, epsilon, T, pi1, pi2, tau):
        c1 = c1_optimal(l, w, p1, p2, alpha, T, pi1, pi2)
        c2 = c2_optimal(l, w, p1, p2, alpha, T, pi1, pi2, tau)
        utility = np.log(c1 ** alpha * c2 ** (1 - alpha)) - nu * (l ** (1 + epsilon)) / (1 + epsilon)
        return -utility  # We minimize the negative utility to maximize utility

    def market_clearing_conditions(prices, par, w):
        p1, p2 = prices
        l1_star = optimal_labor(p1, w, par.A, par.gamma)
        l2_star = optimal_labor(p2, w, par.A, par.gamma)
        y1_star = optimal_output(l1_star, par.A, par.gamma)
        y2_star = optimal_output(l2_star, par.A, par.gamma)
        pi1_star = optimal_profit(w, p1, par.A, par.gamma)
        pi2_star = optimal_profit(w, p2, par.A, par.gamma)
        res = minimize_scalar(
            utility_maximization, 
            bounds=(0, 100), 
            args=(w, p1, p2, par.alpha, par.nu, par.epsilon, par.T, pi1_star, pi2_star, par.tau), 
            method='bounded'
        )
        l_star = res.x
        c1_star = c1_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star)
        c2_star = c2_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star, par.tau)
        labor_clearing = l_star - (l1_star + l2_star)
        good1_clearing = c1_star - y1_star
        return np.array([labor_clearing, good1_clearing]).flatten()

    # The new part to earlier code is this, which finds the equilibrium prices where tau is not already defined, this is used later
    def find_equilibrium_prices_with_tau(par, w, tau, initial_guess=[1.0, 1.0]):
        par.tau = tau
        solution = root(market_clearing_conditions, initial_guess, args=(par, w))
        equilibrium_prices = solution.x
        p1_star, p2_star = equilibrium_prices
        equilibrium_prices_df = pd.DataFrame({
            'p1_star': [p1_star],
            'p2_star': [p2_star]
        })
        return equilibrium_prices_df

    # We define the social welfare function that we want to maximize by minimizing the negative function
    def social_welfare(tau, par, w):
        equilibrium = find_equilibrium_prices_with_tau(par, w, tau)
        p1_eq = equilibrium['p1_star'].values[0]
        p2_eq = equilibrium['p2_star'].values[0]
        l1_star = optimal_labor(p1_eq, w, par.A, par.gamma)
        y1_star = optimal_output(l1_star, par.A, par.gamma)
        pi1_star = optimal_profit(w, p1_eq, par.A, par.gamma)
        l2_star = optimal_labor(p2_eq, w, par.A, par.gamma)
        y2_star = optimal_output(l2_star, par.A, par.gamma)
        pi2_star = optimal_profit(w, p2_eq, par.A, par.gamma)
        total_profits = pi1_star + pi2_star
        par.T = tau * y2_star  # Government budget constraint
        def neg_utility(l):
            return -utility_maximization(l, w, p1_eq, p2_eq, par.alpha, par.nu, par.epsilon, par.T, pi1_star, pi2_star, tau)
        result = minimize(neg_utility, x0=1.0)
        l_star = result.x[0]
        U = -result.fun
        SWF = U - par.kappa * y2_star
        return -SWF  # Minimize the negative of SWF to maximize it

    # We then minimize the social welfare function to find the optimal tau, which we use to find the equilibrium price, which is used to find the optimal output, and at last we find optimal T
    optimal_tau = minimize(social_welfare, x0=0.1, args=(par, w), bounds=[(0, None)])
    tau_opt = optimal_tau.x[0]
    equilibrium = find_equilibrium_prices_with_tau(par, w, tau_opt)
    p2_eq = equilibrium['p2_star'].values[0]
    l2_star = optimal_labor(p2_eq, w, par.A, par.gamma)
    y2_star = optimal_output(l2_star, par.A, par.gamma)
    T_opt = tau_opt * y2_star

    return tau_opt, T_opt