import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import Matrix, lambdify


def calculate_X_positions(indices, N_columns, X_dist):
    """
    Calculate the X positions of the nodes.

    Args:
        indices (list): List or numpy array of indices from 1 to N_nod_tot
        N_columns (int): Number of columns
        X_dist (int): Distance between columns

    Returns:
        List of X positions of the nodes
    """
    X = np.zeros(len(indices))
    X = ((indices % N_columns) - 1) * X_dist
    np.putmask(X, X < 0, (N_columns - 1) * X_dist)
    return X


def calculate_Y_positions(indices, N_columns, Y_dist):
    """
    Calculate the Y positions of the nodes.

    Args:
        indices (list): List or numpy array of indices from 1 to N_nod_tot
        N_columns (int): Number of columns
        Y_dist (int): Distance between columns

    Returns:
        List of Y positions of the nodes
    """
    Y = np.zeros(len(indices))
    h_assigner = np.ceil(indices / N_columns - 1)
    h_assigner[h_assigner < 1] = 0
    Y = Y_dist * h_assigner
    return Y


def calculate_element_node_indices(N_floors, N_columns):
    """
    Calculate the element node indices.

    Args:
        N_floors (int): Number of floors in the frame
        N_columns (int): Number of columns

    Returns:
        Numpy array of element node indices
    """
    # Initialize the ele_nod array
    ele_nod = np.zeros((N_floors * (2 * N_columns - 1), 2), dtype=int)

    # Calculate the indices for the vertical and horizontal elements
    for i in range(1, N_floors + 1):
        for j in range(1, N_columns + 1):
            index = (i - 1) * (2 * N_columns - 1) + j - 1
            ele_nod[index, 0] = j + (i - 1) * N_columns
            ele_nod[index, 1] = ele_nod[index, 0] + N_columns
        for j in range(1, N_columns):
            index = (i - 1) * (2 * N_columns - 1) + N_columns + j - 1
            ele_nod[index, 0] = i * N_columns + j
            ele_nod[index, 1] = ele_nod[index, 0] + 1

    return ele_nod


def calculate_element_length(N_ele_tot, N_columns, X_dist, Y_dist):
    """
    Calculate the length of the elements.

    Args:
        N_ele_tot (int): Number of elements in the frame
        N_columns (int): Number of columns
        X_dist (int): Distance between columns
        Y_dist (int): Height of the columns

    Returns:
        Numpy array of element lengths
    """
    h = np.zeros(N_ele_tot)

    # Calculate h, length of the elements
    for i in range(1, N_ele_tot+1):
        if i % (N_columns+1) != 0:
            h[i-1] = Y_dist
        else:
            h[i-1] = X_dist

    return h


#### Heavy functions
def initialize_symbols(N_par_ele):
    """
    Create and return the symbolic variables used in the calculations.

    Args:
        N_par_ele (int): Number of parameter per element

    Returns:
        Returns a tuple of the symbolic variables
    """
    # Define symbolic variables
    x, h_e, beta_e, beta_curr = sp.symbols('x h_e beta_e beta_curr')
    qe = sp.Array([sp.Symbol(f'q{i}') for i in range(1, N_par_ele+1)])
    a0, a1, c0, c1, c2, c3 = sp.symbols('a0 a1 c0 c1 c2 c3')
    A_e, E_e, J_e, ro_e, T, fo_E = sp.symbols('A_e E_e J_e ro_e T fo_E')
    Qglo_pel_curr1_mode, Qglo_pel_curr2_mode, Qglo_pel_curr3_mode, Qglo_pel_curr4_mode, Qglo_pel_curr5_mode, Qglo_pel_curr6_mode= sp.symbols('Qglo_pel_curr1_mode Qglo_pel_curr2_mode Qglo_pel_curr3_mode Qglo_pel_curr4_mode Qglo_pel_curr5_mode Qglo_pel_curr6_mode')
    X_old, Y_old = sp.symbols('X_old Y_old')

    return (x, h_e, beta_e, beta_curr, qe, a0, a1, c0, c1, c2, c3, A_e, E_e, J_e, ro_e, 
            T, fo_E, Qglo_pel_curr1_mode, Qglo_pel_curr2_mode, Qglo_pel_curr3_mode, 
            Qglo_pel_curr4_mode, Qglo_pel_curr5_mode, Qglo_pel_curr6_mode, X_old, Y_old)


def calculate_energies(x, qe, h_e, beta_e, E_e, J_e, A_e, ro_e, ve_beam, ue_beam):
    """
    Calculate the potential and kinetic energy of the beam.

    Args:
        x (symbol): Symbolic variable for the x-coordinate
        qe (list): List of symbolic variables for the beam parameters
        h_e (float): Height of the beam element
        beta_e (float): Angle of the beam element
        E_e (float): Young's modulus of the beam material
        J_e (float): Polar moment of inertia of the beam cross-section
        A_e (float): Cross-sectional area of the beam
        ro_e (float): Density of the beam material
        ve_beam (sympy expression): Vertical displacement of the beam
        ue_beam (sympy expression): Horizontal displacement of the beam

    Returns:
        Pot_beam (sympy expression): Potential energy of the beam
        Kin_beam (sympy expression): Kinetic energy of the beam
        chi_beam (sympy expression): Curvature of the beam
        eps_beam (sympy expression): Strain of the beam
    """
    # Calculate chi_beam and eps_beam
    chi_beam = sp.diff(sp.diff(ve_beam, x), x)
    eps_beam = sp.diff(ue_beam, x)

    # Calculate potential energy (Pot_beam) and kinetic energy (Kin_beam)
    Pot_beam = 1 / 2 * sp.integrate(E_e * J_e * chi_beam**2 + E_e * A_e * eps_beam**2, (x, 0, h_e))
    Kin_beam = 1 / 2 * ro_e * A_e * sp.integrate(ve_beam**2 + ue_beam**2, (x, 0, h_e))
    
    return Pot_beam, Kin_beam, chi_beam, eps_beam


def calculate_beam_displacement_equations(x, h_e, beta_e, qe, a0, a1, c0, c1, c2, c3):
    """
    Calculate the beam displacement equations.

    Returns:
        Displacement functions for the beam in the u and v directions
    """
    # Compute v1, u1, v2, u2
    v1 = -qe[0] * sp.sin(beta_e) + qe[1] * sp.cos(beta_e)
    u1 = qe[0] * sp.cos(beta_e) + qe[1] * sp.sin(beta_e)
    v2 = -qe[3] * sp.sin(beta_e) + qe[4] * sp.cos(beta_e)
    u2 = qe[3] * sp.cos(beta_e) + qe[4] * sp.sin(beta_e)

    # Define beam displacement equations
    u_beam = a0 + a1 * x 
    v_beam = c0 + c1 * x + c2 * x**2 + c3 * x**3 

    # Define equilibrium equations
    equations = [
        v_beam.subs(x, 0) - v1,
        sp.diff(v_beam, x).subs(x, 0) - qe[2],
        v_beam.subs(x, h_e) - v2,
        sp.diff(v_beam, x).subs(x, h_e) - qe[5],
        u_beam.subs(x, 0) - u1,
        u_beam.subs(x, h_e) - u2
    ]

    # Solve and assign the solution
    sol = sp.solve(equations, (c0, c1, c2, c3, a0, a1))
    ve_beam = v_beam.subs(sol)
    ue_beam = u_beam.subs(sol)

    # Lambdify ve_beam and ue_beam
    ve_beam_func = lambdify((x, qe, h_e, beta_e), ve_beam, "numpy")
    ue_beam_func = lambdify((x, qe, h_e, beta_e), ue_beam, "numpy")

    return ve_beam_func, ue_beam_func, ve_beam, ue_beam


def assemble_global_matrices(N_par_ele, N_par_tot, N_ele_tot, Pot_beam, Kin_beam, qe, h, 
                             A, E, J, beta, ro, pel, h_e, A_e, E_e, J_e, beta_e, ro_e, x):
    """
    Assemble the global stiffness and mass matrices by assembling the 
    symbolic element stiffness and mass matrices and converting them to
    numeric arrays.

    Returns:
        Numeric arrays of the global stiffness and mass matrices
    """
    K_beam = np.zeros((N_par_ele, N_par_ele), dtype=object)
    M_beam = np.zeros((N_par_ele, N_par_ele), dtype=object)

    # Compute K_beam and M_beam
    for i in range(N_par_ele):
        for j in range(N_par_ele):
            K_beam[i][j] = sp.lambdify((x, h_e, A_e, E_e, J_e, beta_e, ro_e),
                                sp.diff(sp.diff(Pot_beam, qe[i]), qe[j]), 'numpy')
            M_beam[i][j] = sp.lambdify((x, h_e, A_e, E_e, J_e, beta_e, ro_e),
                                sp.diff(sp.diff(Kin_beam, qe[i]), qe[j]), 'numpy')

    # Initialize element stiffness matrix (Ke) and global stiffness matrix (K)
    K = np.zeros((N_par_tot, N_par_tot))
    M = np.zeros((N_par_tot, N_par_tot))

    # Compute Ke, Me and assemble K, M using NumPy operations
    for e in range(N_ele_tot):
        for i in range(N_par_ele):
            for j in range(N_par_ele):
                K[pel[e, i]-1, pel[e, j]-1] += K_beam[i, j](0, h[e], A[e], E[e], J[e], beta[e], ro[e])
                M[pel[e, i]-1, pel[e, j]-1] += M_beam[i, j](0, h[e], A[e], E[e], J[e], beta[e], ro[e])

    return K, M


def apply_boundary_conditions(N_par_tot, N_nod_tot, N_par_nod, w, K, M):
    """
    Applies the boundary conditions to the stiffness and mass matrices.

    Returns:
        Numpy arrays of the stiffness and mass matrices with the boundary conditions applied
    """
    mask = w == 1

    K[mask, :] = 0
    K[:, mask] = 0
    M[mask, :] = 0
    M[:, mask] = 0

    # Set the diagonal elements where W is 1 to 1 or 1e-30
    np.fill_diagonal(K, np.where(mask, 1, K.diagonal()))
    np.fill_diagonal(M, np.where(mask, 1e-30, M.diagonal()))

    return K, M


def compute_eigenvalues_and_eigenvectors(K, M):
    """
    Compute the eigenvalues and eigenvectors of the stiffness and mass matrices.


    Returns:
        The real part of the eigenvalues (frequency) and the normalized eigenvectors (modes of vibration)
    """
    # Compute eigenvalues (lamb) and eigenvectors (phis)
    lamb, phis = np.linalg.eig(np.linalg.inv(M) @ K)

    # Get the indices that would sort lamb in descending order
    idx = np.argsort(lamb)[::-1]

    # Sort lamb and phis
    lamb_r = lamb[idx]
    phis_r = phis[:, idx]

    # Normalize eigenvectors
    N_par_tot = len(lamb)
    phis_norm = np.zeros((N_par_tot, N_par_tot))
    for i in range(N_par_tot):
        c = np.sqrt(np.dot(phis_r[:, i].T, M @ phis_r[:, i]))
        phis_norm[:, i] = phis_r[:, i] / c

    return lamb_r, phis_norm


def get_mode_indices(lamb_r, phis_norm, N_plots):
    """
    Calculate the periods to get the top contributing index_modes.

    Returns:
        Numpy array of the indices of the largest N_plots periods
    """
    # Calculate periods
    period = 2 * np.pi / np.sqrt(lamb_r)

    # Find the indices of the largest N_plots periods
    index_modes = np.argpartition(period, -N_plots)[-N_plots:]

    # Sort index_modes so that the modes are in descending order of period
    index_modes = index_modes[np.argsort(period[index_modes])][::-1]

    # Extract lambdas and corresponding eigenvectors (No longer used)
    # lamb_plots = lamb_r[index_modes]
    # phis_plots = phis_norm[:, index_modes]

    return index_modes, period


def calculate_global_displacements(Qglo_pel_curr1_mode, Qglo_pel_curr2_mode, Qglo_pel_curr3_mode, Qglo_pel_curr4_mode, 
                           Qglo_pel_curr5_mode, Qglo_pel_curr6_mode, beta_curr, h_e):
    """
    Calculate the global displacements by solving the local symbolic
    equilibrium equations.

    Returns:
        Lambda functions for the global displacements
    """
    # Define symbols
    x, f0, f1, g0, g1, g2, g3, X_old, Y_old = sp.symbols('x f0 f1 g0 g1 g2 g3 X_old Y_old')

    # Define local displacements
    u_loc_i = Qglo_pel_curr1_mode * sp.cos(beta_curr) + Qglo_pel_curr2_mode * sp.sin(beta_curr)
    v_loc_i = -Qglo_pel_curr1_mode * sp.sin(beta_curr) + Qglo_pel_curr2_mode * sp.cos(beta_curr)
    u_loc_j = Qglo_pel_curr4_mode * sp.cos(beta_curr) + Qglo_pel_curr5_mode * sp.sin(beta_curr)
    v_loc_j = -Qglo_pel_curr4_mode * sp.sin(beta_curr) + Qglo_pel_curr5_mode * sp.cos(beta_curr)

    # Define beam displacements
    u_beam = f1 * x + f0
    v_beam = g3 * x**3 + g2 * x**2 + g1 * x + g0

    # Define equilibrium equations
    equations = [
        v_beam.subs(x, 0) - v_loc_i,
        sp.diff(v_beam, x).subs(x, 0) - Qglo_pel_curr3_mode,
        v_beam.subs(x, h_e) - v_loc_j,
        sp.diff(v_beam, x).subs(x, h_e) - Qglo_pel_curr6_mode,
        u_beam.subs(x, 0) - u_loc_i,
        u_beam.subs(x, h_e) - u_loc_j
    ]

    # Solve the equations
    sol = sp.solve(equations, (f0, f1, g0, g1, g2, g3))

    # Assign the solution
    f0, f1, g0, g1, g2, g3 = sol.values()
    u_beam = u_beam.subs(sol)
    v_beam = v_beam.subs(sol)


    # Define new coordinates using the same expressions as in the original code
    X_new_expr = X_old + x * sp.cos(beta_curr) + u_beam * sp.cos(beta_curr) - v_beam * sp.sin(beta_curr)
    Y_new_expr = Y_old + x * sp.sin(beta_curr) + u_beam * sp.sin(beta_curr) + v_beam * sp.cos(beta_curr)

    # Substitute the solution into the expressions
    X_new_expr_sub = X_new_expr.subs(sol)
    Y_new_expr_sub = Y_new_expr.subs(sol)

    # Convert X_new and Y_new to lambda functions
    X_new_sub_func = lambdify(
        (x, X_old, Y_old, beta_curr, Qglo_pel_curr1_mode, Qglo_pel_curr2_mode, Qglo_pel_curr3_mode, Qglo_pel_curr4_mode, Qglo_pel_curr5_mode, Qglo_pel_curr6_mode, h_e),
        X_new_expr_sub,
        "numpy",
    )

    Y_new_sub_func = lambdify(
        (x, X_old, Y_old, beta_curr, Qglo_pel_curr1_mode, Qglo_pel_curr2_mode, Qglo_pel_curr3_mode, Qglo_pel_curr4_mode, Qglo_pel_curr5_mode, Qglo_pel_curr6_mode, h_e),
        Y_new_expr_sub,
        "numpy",
    )

    return X_new_sub_func, Y_new_sub_func


def print_matrix(matrix, width=8, precision=3, row_labels=None, col_labels=None):
    if row_labels is None:
        row_labels = range(1, matrix.shape[0] + 1)
    if col_labels is None:
        col_labels = range(1, matrix.shape[1] + 1)

    # Header row
    print(" " * width, end="")
    for label in col_labels:
        print(f"{label:{width}}", end="")
    print()

    # Matrix rows
    for i, row in enumerate(matrix):
        print(f"{row_labels[i]:{width}}", end="")
        for val in row:
            print(f"{val:{width}.{precision}f}", end="")
        print()