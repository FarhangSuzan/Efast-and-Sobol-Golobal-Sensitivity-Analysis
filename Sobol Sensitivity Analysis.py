from scipy.integrate import odeint
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt

#Replace your own Model
def system(Y, t, k, c, mu, beta, e, delta_Ic, delta_In, delta_Im, d_I, phi, d_d, delta_dm, alpha_Im, d_m, mu_n, delta_n, gamma_n, xi, d_f, zeta, d_Il, rho_c, delta_CIl, d_c, rho_a, d_a):
    V, I1, I2, T, R, D, M_phi, N_k, F, Il, C, A = Y

    dVdt =
    dI1dt =
    dI2dt =
    dTdt =
    dRdt =
    dDdt =
    dM_phidt =
    dN_kdt =
    dFdt =
    dIldt =
    dCdt =
    dAdt =

    return [dVdt, dI1dt, dI2dt, dTdt, dRdt, dDdt, dM_phidt, dN_kdt, dFdt, dIldt, dCdt, dAdt]
#************************
problem = {
    'num_vars': 29,
    'names': ['k', 'c', 'mu', 'beta', 'e', 'delta_Ic', 'delta_In', 'delta_Im', 'd_I', 'phi', 'd_d', 'delta_dm', 'alpha_Im', 'd_m', 'mu_n', 'delta_n', 'gamma_n', 'xi', 'd_f', 'zeta', 'd_Il', 'rho_c', 'delta_CIl', 'd_c', 'rho_a', 'd_a', 'I0', 'V0', 'dummy'],
    'bounds': [
        [0, 1],         # k
        [0.1, 0.9],     # c
        [0, 0.5],       # mu
        [0.1, 0.6],     # beta
        [0, 0.7],       # e
        [0, 0.5],       # delta_Ic
        [0, 0.4],       # delta_In
        [0.2, 0.6],     # delta_Im
        [0.1, 0.3],     # d_I
        [0, 1],         # phi
        [0, 0.4],       # d_d
        [0.1, 0.5],     # delta_dm
        [0.2, 0.7],     # alpha_Im
        [0.2, 0.6],     # d_m
        [0, 0.4],       # mu_n
        [0, 0.5],       # delta_n
        [0, 0.3],       # gamma_n
        [0.1, 0.4],     # xi
        [0.2, 0.7],     # d_f
        [0.1, 0.5],     # zeta
        [0, 0.4],       # d_Il
        [0.2, 0.6],     # rho_c
        [0, 0.5],       # delta_CIl
        [0.1, 0.4],     # d_c
        [0.2, 0.7],     # rho_a
        [0, 0.3],       # d_a
        [0, 0.1],       # I0
        [0.9, 1],       # V0
        [0, 1]          #dummy
    ]
}

# Generate samples
param_values = saltelli.sample(problem, 1024)  # Using Saltelli's sampling method
####################################################
# Modify the evaluate_model function to take an additional argument for the variable index
def evaluate_model(params, var_idx):
    Y0 = [params[-2], params[-3], 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # Adjust initial conditions if necessary
    timespan = np.linspace(0, 10, 100)  # Arbitrary timespan

    # Exclude the dummy parameter by using params[:-3] instead of params[:-2]
    solution = odeint(system, Y0, timespan, args=tuple(params[:-3]))

    return solution[-1, var_idx]

# Variables' names (for plotting)
variables_names = ['V', 'I1', 'I2', 'T', 'R', 'D', 'M_phi', 'N_k', 'F', 'Il', 'C', 'A']

for idx, var_name in enumerate(variables_names):
    # Run the model for each sample and selected variable
    output = np.array([evaluate_model(params, idx) for params in param_values])

    # Analyze sensitivity
    Si = sobol.analyze(problem, output)

    # Plotting First-order Sensitivities for the variable
    plt.figure()
    plt.bar(range(len(problem['names'])), Si['S1'])
    plt.xticks(range(len(problem['names'])), problem['names'], rotation='vertical')
    plt.ylabel("First Order Sensitivity Index")
    plt.title(f"First Order Sensitivity Analysis for {var_name}")
    plt.tight_layout()
    plt.show()

    # Plotting Total-order Sensitivities for the variable
    plt.figure()
    plt.bar(range(len(problem['names'])), Si['ST'])
    plt.xticks(range(len(problem['names'])), problem['names'], rotation='vertical')
    plt.ylabel("Total Order Sensitivity Index")
    plt.title(f"Total Order Sensitivity Analysis for {var_name}")
    plt.tight_layout()
    plt.show()
