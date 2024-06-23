import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

def simulate_expected_utility(par):
    # We set the seed to 2000 to be able to reproduce the results.
    np.random.seed(2000)  
    # We start off by creating an array of zeros to store the utilities for each career.
    utilities = np.zeros(par.J)
    # We then loop over the number of careers and calculate the expected utility for each career.
    for j in range(par.J):
        epsilon = np.random.normal(0, par.sigma, par.K)
        utilities[j] = par.v[j] + np.mean(epsilon)
    return utilities

def simulate_career_choices(par):
    # We set the seed to 2000 to be able to reproduce the results.
    np.random.seed(2000)  
    # We set up arrays of zeros to store the chosen careers, expected utilities, and realized utilities.
    chosen_careers = np.zeros((par.N, par.K), dtype=int)
    expected_utilities = np.zeros((par.N, par.K))
    realized_utilities = np.zeros((par.N, par.K))
    # We then want to iterate over each graduate and calculate the career choice, expected utility, and realized utility.
    for k in range(par.K):
        for i in range(par.N):
            Fi = par.F[i]
            prior_expected_utility = np.zeros(par.J)
            own_noise = np.random.normal(0, par.sigma, par.J)
            for j in range(par.J):
                # Here we generate the noise for the friends of the graduate.
                friends_noise = np.random.normal(0, par.sigma, Fi)
                prior_expected_utility[j] = par.v[j] + np.mean(friends_noise)
            # We set it up so the graduate chooses the career with the highest expected utility.
            chosen_career = np.argmax(prior_expected_utility)
            chosen_careers[i, k] = chosen_career
            expected_utilities[i, k] = prior_expected_utility[chosen_career]
            # At last we can calculate and store the realized utility for the chosen career.
            realized_utilities[i, k] = par.v[chosen_career] + own_noise[chosen_career]
    return chosen_careers, expected_utilities, realized_utilities

def calculate_and_prepare_figures(par, chosen_careers, expected_utilities, realized_utilities):
    # We set up a function that makes us able to plot the results.
    # First off, we create arrays to store the career shares, average expected utilities, and average realized utilities.
    career_shares = np.zeros((par.N, par.J))
    avg_expected_utilities = np.zeros(par.N)
    avg_realized_utilities = np.zeros(par.N)
    # We then interate over the different graduate types and calculate the career shares, average expected utilities, and average realized utilities.
    for i in range(par.N):
        for j in range(par.J):
            career_shares[i, j] = np.mean(chosen_careers[i] == j)
        # We now calculate the average expected utility and average realized utility for each graduate type.
        avg_expected_utilities[i] = np.mean(expected_utilities[i])
        avg_realized_utilities[i] = np.mean(realized_utilities[i])
    figures = []
    # We set up the plot of the career shares
    fig1, ax1 = plt.subplots()
    for j in range(par.J):
        ax1.plot(par.F, career_shares[:, j], label=f'Career {j+1}')
    ax1.set_xlabel('Number of Friends (F_i)')
    ax1.set_ylabel('Share of Graduates Choosing Career')
    ax1.legend()
    ax1.set_title('Share of Graduates Choosing Each Career')
    figures.append(fig1)
    # Now we set up the plot showing average expected utility
    fig2, ax2 = plt.subplots()
    ax2.plot(par.F, avg_expected_utilities, label='Average Expected Utility')
    ax2.set_xlabel('Number of Friends (F_i)')
    ax2.set_ylabel('Utility')
    ax2.set_title('Average Subjective Expected Utility of Graduates')
    figures.append(fig2)
    # At last we set up the plot for average realized utility
    fig3, ax3 = plt.subplots()
    ax3.plot(par.F, avg_realized_utilities, label='Average Realized Utility')
    ax3.set_xlabel('Number of Friends (F_i)')
    ax3.set_ylabel('Utility')
    ax3.set_title('Average Realized Utility of Graduates')
    figures.append(fig3)
    return figures

def simulate_second_year_choices(par):
    # We use the same seed as ealiear to be able to reproduce the results.
    np.random.seed(2000)  
    # As for the earlier questions, we set up arrays of zeros, but this time we store the choices and utilities for both years. 
    chosen_careers = np.zeros((par.N, par.K), dtype=int)
    expected_utilities = np.zeros((par.N, par.K))
    realized_utilities = np.zeros((par.N, par.K))
    new_chosen_careers = np.zeros((par.N, par.K), dtype=int)
    new_expected_utilities = np.zeros((par.N, par.K))
    new_realized_utilities = np.zeros((par.N, par.K))
    switch_decision = np.zeros((par.N, par.K), dtype=bool)
    # As before, we iterate over the graduates and calculate the career choices, expected utilities, and realized utilities for both years.
    for k in range(par.K):
        for i in range(par.N):
            Fi = par.F[i]
            prior_expected_utility = np.zeros(par.J)
            own_noise = np.random.normal(0, par.sigma, par.J)
            for j in range(par.J):
                friends_noise = np.random.normal(0, par.sigma, Fi)
                prior_expected_utility[j] = par.v[j] + np.mean(friends_noise)
            # We now use numpy to find the first year career choice
            chosen_career = np.argmax(prior_expected_utility)
            chosen_careers[i, k] = chosen_career
            expected_utilities[i, k] = prior_expected_utility[chosen_career]
            # We calculate the first year realized utility for the chosen career and store it.
            realized_utilities[i, k] = par.v[chosen_career] + own_noise[chosen_career]
            # Again we use numpy, but this time to find the second year career choice
            second_year_prior_utility = np.zeros(par.J)
            # We then calculate the expected utility for the second year for each career.
            for j in range(par.J):
                if j == chosen_career:
                    second_year_prior_utility[j] = realized_utilities[i, k]
                else:
                    friends_noise = np.random.normal(0, par.sigma, Fi)
                    second_year_prior_utility[j] = par.v[j] + np.mean(friends_noise) - par.c
            new_chosen_career = np.argmax(second_year_prior_utility)
            new_chosen_careers[i, k] = new_chosen_career
            new_expected_utilities[i, k] = second_year_prior_utility[new_chosen_career]
            new_realized_utilities[i, k] = par.v[new_chosen_career] + own_noise[new_chosen_career]
            # We create an if statement to check if the graduate switches career, and if they do, we remove the parameter c from their utility.
            if new_chosen_career != chosen_career:
                new_realized_utilities[i, k] -= par.c
                switch_decision[i, k] = True
    return chosen_careers, new_chosen_careers, new_expected_utilities, new_realized_utilities, switch_decision

# We then set up a function that calculates and visualizes the results from the earlier function
def calculate_and_visualize(par, chosen_careers, new_chosen_careers, new_expected_utilities, new_realized_utilities, switch_decision):
    new_career_shares = np.zeros((par.N, par.J))
    new_avg_expected_utilities = np.zeros(par.N)
    new_avg_realized_utilities = np.zeros(par.N)
    switch_shares = np.zeros((par.N, par.J))
    for i in range(par.N):
        for j in range(par.J):
            new_career_shares[i, j] = np.mean(new_chosen_careers[i] == j)
            switch_shares[i, j] = np.mean(switch_decision[i] & (chosen_careers[i] == j))
        new_avg_expected_utilities[i] = np.mean(new_expected_utilities[i])
        new_avg_realized_utilities[i] = np.mean(new_realized_utilities[i])
    figures = []
    # We plot new career shares
    fig1, ax1 = plt.subplots()
    for j in range(par.J):
        ax1.plot(par.F, new_career_shares[:, j], label=f'Career {j+1}')
    ax1.set_xlabel('Number of Friends (F_i)')
    ax1.set_ylabel('Share of Graduates Choosing Career (Second Year)')
    ax1.legend()
    ax1.set_title('Share of Graduates Choosing Each Career (Second Year)')
    figures.append(fig1)
    # We plot average expected utility (Second Year)
    fig2, ax2 = plt.subplots()
    ax2.plot(par.F, new_avg_expected_utilities, label='Average Expected Utility (Second Year)')
    ax2.set_xlabel('Number of Friends (F_i)')
    ax2.set_ylabel('Utility')
    ax2.set_title('Average Subjective Expected Utility of Graduates (Second Year)')
    figures.append(fig2)
    # Then we plot average realized utility (Second Year)
    fig3, ax3 = plt.subplots()
    ax3.plot(par.F, new_avg_realized_utilities, label='Average Realized Utility (Second Year)')
    ax3.set_xlabel('Number of Friends (F_i)')
    ax3.set_ylabel('Utility')
    ax3.set_title('Average Realized Utility of Graduates (Second Year)')
    figures.append(fig3)
    # At last we plot switch shares
    fig4, ax4 = plt.subplots()
    for j in range(par.J):
        ax4.plot(par.F, switch_shares[:, j], label=f'Switched from Career {j+1}')
    ax4.set_xlabel('Number of Friends (F_i)')
    ax4.set_ylabel('Share of Graduates Switching Careers')
    ax4.legend()
    ax4.set_title('Share of Graduates Switching Careers (Second Year)')
    figures.append(fig4)
    return figures