import numpy as np
import scipy.stats as stats
from scipy.integrate import solve_ivp
from SALib.sample import fast_sampler
from SALib.analyze import fast
import matplotlib.pyplot as plt

# Setting Custom Font Style
csfont = {'fontname': 'Arial'}

# Defining the Model Function (Our ODEs)
#Replace your own ODE Equations
def model(t, y, params):
    V, T, R, I1, I2, F, C, A, M, Il, Nk = y
    k, c, mu, beta, e, d_I, delta_Ic, delta_Im, delta_In, phi, alpha_Im, d_m, xi, d_f, rho_c, d_c, rho_a, d_a, d_r, zeta, d_Il, mu_n, delta_n, gamma_n, dummy = params

    dVdt =
    dTdt =
    dRdt =
    dI1dt =
    dI2dt =
    dFdt =
    dCdt =
    dAdt =
    dMdt =
    dIldt =
    dNkdt = 

    return [dVdt, dTdt, dRdt, dI1dt, dI2dt, dFdt, dCdt, dAdt, dMdt, dIldt, dNkdt]

# Initial conditions
y0 = [1e-10, 0.16, 0, 0, 0, 0.015, 1e-7, 5e-6, 1e-5, 1.1, 387E-6]

# Time points
t = np.linspace(0, 8.247, 100)

# # Problem definition for EFAST
problem = {
    'num_vars': 25,
            'names': [
                'k', 'c', r'$\mu$', r'$\beta$', r'$e$', r'$d_I$', r'$\delta_{Ic}$', r'$\delta_{Im}$', r'$\delta_{In}$',
                r'$\phi$', r'$\alpha_{Im}$', r'$d_m$', r'$\xi$', r'$d_f$', r'$\rho_c$', r'$d_c$', r'$\rho_a$', r'$d_a$', r'$d_r$',
                r'$\zeta$', r'$d_{Il}$', r'$\mu_n$', r'$\delta_n$', r'$\gamma_n$', 'dummy'
            ],
    'bounds': [
        [0.5*22.71, 1.5*22.71],         # 1:k=22.71
        [0.5*1.81, 1.5*1.81],           # 2:c=1.81
        [0.5*1, 1.5*1],                 # 3:mu=0.5
        [0.5*5.681, 1.5*5.681],         # 4:beta=5.681
        [0.5*7.88, 1.5*7.88],           # 5:e=7.88
        [0.5*0.014, 1.5*0.014],         # 6:d_I=0.014
        [0.5*238, 1.5*238],             # 7:delta_Ic=238
        [0.5*121.195, 1.5*121.195],     # 8:delta_Im=121.195
        [0.5*200, 1.5*200],             # 9:delta_In=200
        [0.5*0.14, 1.5*0.14],           # 10:phi=0.14
        [0.5*1.1E+3, 1.5*1.1E+3],       # 11:alpha_Im=1.1E+3
        [0.5*0.3, 1.5*0.3],             # 12:d_m=0.3
        [0.5*60, 1.5*60],               # 13:xi=60
        [0.5*2, 1.5*2],                 # 14:d_f=2
        [0.5*250, 1.5*250],             # 15:rho_c=250
        [0.5*0.01, 1.5*0.01],           # 16:d_c=0.01
        [0.5*80, 1.5*80],               # 17:rho_a=80
        [0.5*0.06, 1.5*0.06],           # 18:d_a=0.06
        [0.5*0.05, 1.5*0.05],           # 19:d_r=0.05
        [0.5*0.5, 1.5*0.5],             #20:'zeta'
        [0.5*2, 1.5*2],                 #21:'d_Il'
        [0.5*0.52, 1.5*0.52],           #22:'mu_n'
        [0.5*0.5, 1.5*0.5],             #23:'delta_n'
        [0.5*0.0007, 1.5*0.0007],        #24:'gamma_n'
        [0, 1]  #25: Bounds for dummy parameter (value does not matter)
    ]
}

# Wrapper function for the model
# A function to solve the ODEs for a given set of parameters. It returns the final state of the system if successful; otherwise, it prints an error and returns NaN
def evaluate_model(params):
    sol = solve_ivp(lambda t, y: model(t, y, params), [t[0], t[-1]], y0, t_eval=t, method='LSODA', vectorized=True, atol=1e-6, rtol=1e-3)
    if sol.success:
        return sol.y[:, -1]
    else:
        print("Integration failed for parameters:", params)
        return np.full(len(y0), np.nan)  # Return NaN if the integration fails

# Sampling Parameters
param_values = fast_sampler.sample(problem, 4 * 25**2)#Generates samples of parameters using the EFAST method. The number of samples is 4 * (number of variables)^2

# Evaluating model for each parameter set
outputs = np.array([evaluate_model(params) for params in param_values])#Evaluates the model for each set of sampled parameters

# Handle NaN values in outputs
outputs = np.nan_to_num(outputs, nan=0.0)#Converts NaN values in the outputs to zero, which is necessary for proper sensitivity analysis

# Perform the sensitivity analysis for each output variable
# For each output variable, performs sensitivity analysis using the EFAST method and stores the results in Si_list
Si_list = []
for i in range(len(y0)):
    Si = fast.analyze(problem, outputs[:, i], print_to_console=False)
    Si_list.append(Si)

# Plotting function for EFAST results with error bars and significance
def plot_efast_results_for_all_variables(Si_list, param_names, variables):
    n_vars = len(variables)
    n_params = len(param_names)

    for var_index, var_name in enumerate(variables):
        plt.figure(figsize=(10, 8))
        S1_values = [Si_list[var_index]['S1'][i] for i in range(n_params)]
        ST_values = [Si_list[var_index]['ST'][i] for i in range(n_params)]
        barWidth = 0.5
        r1 = np.arange(len(S1_values))
        r2 = [x + barWidth for x in r1]
        plt.bar(r1, S1_values, width=barWidth, label='First Order (S1)', color='#3C486B')
        plt.bar(r2, ST_values, width=barWidth, label='Total Order (ST)', color='#F45050')
        plt.xlabel('Parameters', fontsize=30)
        plt.ylabel('Sensitivity Index', fontsize=30)
        plt.title(f'Sensitivity Analysis for {var_name}', fontsize=30)
        plt.xticks([r + barWidth / 2 for r in range(len(S1_values))], param_names, rotation=45, ha="right")
        plt.legend()
        plt.tick_params(axis='y', which='major', labelsize=20, direction="in")  # For y-axis
        plt.tick_params(axis='x', which='major', labelsize=12, direction="in")  # For x-axis
        plt.ylim(0, 1)
        # plt.grid(True, which='both', linestyle='--')
        plt.show()

# Call the plotting function
variables = ['V','T','R', r'$I_1$', r'$I_2$', 'IFN', 'C', 'A', r'$M_{\phi}$', r'$IL$', 'Nk']
plot_efast_results_for_all_variables(Si_list, problem['names'], variables)

plt.show()
