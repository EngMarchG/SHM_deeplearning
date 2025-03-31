import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import Matrix, lambdify
<<<<<<< HEAD
from constraints import N_columns, N_floors, N_nod_tot, N_par_nod, N_par_tot, N_ele_tot, N_nod_ele, N_par_ele, N_tot_bound, N_plots
from constraints import X_dist, Y_dist, width_beam, height_beam, width_column, height_column, po, theta, unit_weight, elastic_mod, N_discritizations
from frame_helpers import *

# Options
TO_PLOT = True
PLOT_EMPHISIS = 1
=======
from utils.frame_constraints import (
    N_columns, N_floors, N_nod_tot, N_par_nod, N_par_tot, N_ele_tot, 
    N_nod_ele, N_par_ele, N_tot_bound, N_plots, X_dist, Y_dist, 
    width_beam, height_beam, width_column, height_column, po, theta, 
    unit_weight, elastic_mod, N_discritizations
)
from utils.frame_helpers import *

# Options
TO_PLOT = True
PLOT_EMPHISIS = 3 # Makes the plot deformations more pronounced
>>>>>>> main


# Global position of each element
indices = np.arange(1, N_nod_tot + 1)  # array of indices from 1 to N_nod_tot

# Calculate X positions
X = calculate_X_positions(indices, N_columns, X_dist)

# Calculate Y positions
Y = calculate_Y_positions(indices, N_columns, Y_dist)


# -------- BOUNDARY CONDITIONS --------
# For non ordered bounds, must be changed.
w = np.zeros((N_nod_tot, N_par_nod))
w[:N_columns, :] = 1

# Initialize matrices
W = np.zeros(N_par_tot, dtype=int)
<<<<<<< HEAD
count = 0

# W = 1 if the degree of freedom is fixed, 0 if it is free
=======

# W = 1 if the degree of freedom is fixed, 0 if it is free
count = 0
>>>>>>> main
for i in range(N_nod_tot):
    for j in range(N_par_nod):
        W[count] = w[i, j]
        count += 1


# -------- ELEMENT NODE INDICIES (ele_nod[element,node]) --------
ele_nod = calculate_element_node_indices(N_floors, N_columns)


# -------- PAR MATRIX AND PEL MATRIX --------
# Initialize matrices
par = np.arange(1, N_nod_tot * N_par_nod + 1).reshape(N_nod_tot, N_par_nod).astype(int)
pel = np.zeros((N_ele_tot, 2 * N_par_nod), dtype=int)

# Assign values to pel matrix
for i in range(N_ele_tot):
    pel[i, :N_par_nod] = par[ele_nod[i, 0] - 1, :]
    pel[i, N_par_nod:] = par[ele_nod[i, 1] - 1, :]


# -------- ELEMENT CHARACTERIZING DATA --------
J = np.zeros(N_ele_tot)
A = np.zeros(N_ele_tot)
beta = np.zeros(N_ele_tot)
E = np.array([elastic_mod] * N_ele_tot, dtype=float)
ro = np.array([unit_weight] * N_ele_tot, dtype=float)

# Calculate h, length of the elements
h = calculate_element_length(N_ele_tot, N_columns, X_dist, Y_dist)

# Calculate beta
for i in range(N_ele_tot):
    beta[i] = np.arccos((X[ele_nod[i, 1] - 1] - X[ele_nod[i, 0] - 1]) / h[i])


# Calculate J, A, and h
<<<<<<< HEAD
count = 0
for i in range(N_floors):
    for j in range(N_columns):
        J[count] = width_column * height_column**3 / 12
        A[count] = width_column * height_column
        h[count] = Y_dist
        E[count] = 21*10**7
        count += 1
    for j in range(N_columns - 1):
        J[count] = width_beam**3 * height_beam / 12
        A[count] = width_beam * height_beam
        h[count] = X_dist
        count += 1
=======
# Total elements per floor = N_columns (columns) + (N_columns - 1) (beams)
elements_per_floor = 2 * N_columns - 1

# Create a mask for column elements: True for first N_columns in every floor, False for beams.
col_mask = np.tile(np.concatenate((np.ones(N_columns, dtype=bool),
                                     np.zeros(N_columns - 1, dtype=bool))),
                   N_floors)
beam_mask = ~col_mask

J[col_mask] = width_column * height_column**3 / 12
A[col_mask] = width_column * height_column
h[col_mask] = Y_dist
E[col_mask] = 21 * 10**7

J[beam_mask] = width_beam**3 * height_beam / 12
A[beam_mask] = width_beam * height_beam
h[beam_mask] = X_dist
E[beam_mask] = 21 * 10**10
>>>>>>> main


# Define symbolic variables
(x, h_e, beta_e, beta_curr, qe, a0, a1, c0, c1, c2, c3, A_e, E_e, J_e, ro_e, T, 
fo_E, Qglo_pel_curr1_mode, Qglo_pel_curr2_mode, Qglo_pel_curr3_mode, 
Qglo_pel_curr4_mode, Qglo_pel_curr5_mode, Qglo_pel_curr6_mode, X_old, Y_old) = initialize_symbols(N_par_ele)

# Calculate the symbolic displacements
ve_beam_func, ue_beam_func, ve_beam, ue_beam  = calculate_beam_displacement_equations(x, h_e, beta_e, qe, a0, a1, c0, c1, c2, c3)


<<<<<<< HEAD

### Unsed for response solutions
=======
# -------- Unsed for response solutions --------
>>>>>>> main
# Define external force array (fo)
fo = sp.zeros(N_ele_tot, 1)
for i in range(0, N_ele_tot, 3):
    fo[i] = -po * sp.sin(theta * T)

# Calculate external force (ext_f)
ext_f = sp.integrate(fo_E * ve_beam, (x, 0, h_e))

# Initialize external force vector (p_E)
p_E = Matrix.zeros(N_par_tot, 1)
for i in range(N_par_ele):
    p_E[i] = sp.diff(ext_f, qe[i]).evalf()

# Initialize global force vector (P)
P = np.zeros(N_par_tot)
<<<<<<< HEAD


=======
# -------- Unsed for response solutions --------


# -------- CALCULATE ENERGIES --------
>>>>>>> main
# Calculate potential energy (Pot_beam) and kinetic energy (Kin_beam)
Pot_beam, Kin_beam, chi_beam, eps_beam = calculate_energies(x, qe, h_e, beta_e, E_e, J_e, A_e, ro_e, ve_beam, ue_beam)



# -------- ASSEMBLE GLOBAL STIFFNESS MATRIX --------
# Initialize K_beam and M_beam as SymPy matrices
K, M = assemble_global_matrices(N_par_ele, N_par_tot, N_ele_tot, Pot_beam, Kin_beam, qe, h, A, E, J, beta, ro, pel, h_e, A_e, E_e, J_e, beta_e, ro_e, x)
K, M = apply_boundary_conditions(N_par_tot, N_nod_tot, N_par_nod, W, K, M)



# -------- EIGENVALUE PROBLEM --------
# Compute eigenvalues (lamb) and eigenvectors (phis)
# Already taken the real part and normalized
lamb_r, phis_norm = compute_eigenvalues_and_eigenvectors(K, M)


# Calculate periods to get the top contributing index_modes
index_modes, period = get_mode_indices(lamb_r, phis_norm, N_plots)

# Extract lambdas and corresponding eigenvectors
lamb_plots = lamb_r[index_modes]
phis_plots = phis_norm[:, index_modes]


# -------- Finding the global displacement --------
# Define local displacements
X_new_sub_func, Y_new_sub_func = calculate_global_displacements(Qglo_pel_curr1_mode, Qglo_pel_curr2_mode, Qglo_pel_curr3_mode, Qglo_pel_curr4_mode, 
                           Qglo_pel_curr5_mode, Qglo_pel_curr6_mode, beta_curr, h_e)



# Define lists
X_new_sub = np.zeros((N_plots, N_ele_tot, N_discritizations))
Y_new_sub = np.zeros((N_plots, N_ele_tot, N_discritizations))

# Initialize X_disp
X_disp = np.zeros((N_plots, N_ele_tot))
Y_disp = np.zeros((N_plots, N_ele_tot))

<<<<<<< HEAD
# Iterate over N_plots and N_ele_tot
for j in range(N_plots):
    for e in range(N_ele_tot):
        # Substitute values into X_new and Y_new using the lambda functions
        substitutions = {
            Qglo_pel_curr1_mode: phis_plots[pel[e, 0] - 1, j]*PLOT_EMPHISIS,
            Qglo_pel_curr2_mode: phis_plots[pel[e, 1] - 1, j]*PLOT_EMPHISIS,
            Qglo_pel_curr3_mode: phis_plots[pel[e, 2] - 1, j]*PLOT_EMPHISIS,
            Qglo_pel_curr4_mode: phis_plots[pel[e, 3] - 1, j]*PLOT_EMPHISIS,
            Qglo_pel_curr5_mode: phis_plots[pel[e, 4] - 1, j]*PLOT_EMPHISIS,
            Qglo_pel_curr6_mode: phis_plots[pel[e, 5] - 1, j]*PLOT_EMPHISIS,
=======
if TO_PLOT:
    # Create a figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Iterate over N_plots and N_ele_tot, saving displacements and plotting on subplots
for j in range(N_plots):
    # Reset flags for legend so each subplot gets one entry per category
    if TO_PLOT:
        ax = axs[j // 2, j % 2]
        columns_plotted = False
        beams_plotted = False

    for e in range(N_ele_tot):
        # Build substitution dictionary for the current element and mode
        substitutions = {
            Qglo_pel_curr1_mode: phis_plots[pel[e, 0] - 1, j] * PLOT_EMPHISIS,
            Qglo_pel_curr2_mode: phis_plots[pel[e, 1] - 1, j] * PLOT_EMPHISIS,
            Qglo_pel_curr3_mode: phis_plots[pel[e, 2] - 1, j] * PLOT_EMPHISIS,
            Qglo_pel_curr4_mode: phis_plots[pel[e, 3] - 1, j] * PLOT_EMPHISIS,
            Qglo_pel_curr5_mode: phis_plots[pel[e, 4] - 1, j] * PLOT_EMPHISIS,
            Qglo_pel_curr6_mode: phis_plots[pel[e, 5] - 1, j] * PLOT_EMPHISIS,
>>>>>>> main
            X_old: X[ele_nod[e, 0] - 1],
            Y_old: Y[ele_nod[e, 0] - 1],
            beta_curr: beta[e],
            h_e: h[e],
        }

<<<<<<< HEAD
        # Save the displacement [# of element, # of mode]
        X_new_sub[j][e] = X_new_sub_func(np.linspace(0, h[e], 
                                                     N_discritizations), **{str(k): v for k, v in substitutions.items()})
        X_disp[j, e] = X_new_sub[j][e][-1] - X[ele_nod[e, 1]-1]
        
        Y_new_sub[j][e] = Y_new_sub_func(np.linspace(0, h[e], 
                                                     N_discritizations), **{str(k): v for k, v in substitutions.items()})
        Y_disp[j, e] = Y_new_sub[j][e][-1] - Y[ele_nod[e, 1]-1]

        # print(f"Mode: {j}, Element: {e}, X_disp: {X_disp[j, e]}")



        if TO_PLOT:
            # Plot the displacement of the current mode
            plt.plot(X_new_sub[j][e], Y_new_sub[j][e])
    if TO_PLOT:
        plt.axis('equal')  # Make x and y axes have the same scale
        plt.show()  # Display the plot for the current mode
        

print(X_disp[0,:])
=======
        # Create a discretized x range along the element's length
        x_range = np.linspace(0, h[e], N_discritizations)
        # Save the displacement curves for the current element and mode
        X_new_sub[j][e] = X_new_sub_func(x_range, **{str(k): v for k, v in substitutions.items()})
        Y_new_sub[j][e] = Y_new_sub_func(x_range, **{str(k): v for k, v in substitutions.items()})
        
        # Calculate end-point displacements (difference from the original nodal positions)
        X_disp[j, e] = X_new_sub[j][e][-1] - X[ele_nod[e, 1] - 1]
        Y_disp[j, e] = Y_new_sub[j][e][-1] - Y[ele_nod[e, 1] - 1]

        if TO_PLOT:
            # Plot using blue for columns and red for beams (only one legend entry per type)
            if col_mask[e]:
                if not columns_plotted:
                    ax.plot(X_new_sub[j][e], Y_new_sub[j][e], color='blue', label='Columns')
                    columns_plotted = True
                else:
                    ax.plot(X_new_sub[j][e], Y_new_sub[j][e], color='blue')
            else:
                if not beams_plotted:
                    ax.plot(X_new_sub[j][e], Y_new_sub[j][e], color='red', label='Beams')
                    beams_plotted = True
                else:
                    ax.plot(X_new_sub[j][e], Y_new_sub[j][e], color='red')

    if TO_PLOT:
        # Adjust subplot settings: equal axes, title, and legend
        ax.axis('equal')
        ax.set_title(f"Mode {j+1}")
        ax.legend()

if TO_PLOT:
    plt.tight_layout()
    plt.show()

# Finally, print the saved displacements for the first mode for verification
print(X_disp[0, :])
>>>>>>> main
