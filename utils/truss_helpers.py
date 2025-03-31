import logging
import numpy as np
import sympy as sp
import scipy.linalg
from typing import Tuple, List, Optional
from sympy import Matrix, lambdify
import multiprocessing as mp

# Get the logger
logger = logging.getLogger(__name__)

################## Basic Functions ##################
def initialize_symbols(n_par_ele: int) -> Tuple:
    """
    Create and return the symbolic variables used in the calculations.

    Args:
        n_par_ele: Number of parameter per element

    Returns:
        Returns a tuple of the symbolic variables
    """
    # Define symbolic variables
    x, xi, h_e, beta_e, beta_curr = sp.symbols('x xi h_e beta_e beta_curr')
    A_e, E_e, J_e, ro_e, T, fo_E = sp.symbols('A_e E_e J_e ro_e T fo_E')
    qe = sp.symbols(f'qe:{n_par_ele}')

    a_arr = sp.symbols('a:2') # for axial
    b_arr = sp.symbols('b:2') # Rods
    d_arr = sp.symbols('d:2') # Rods
    c_arr = sp.symbols('c:4') # transversal
    e_arr = sp.symbols('e:3') # timoshenko

    # For global displacements
    Qglo_pel_curr = sp.symbols(f'Qglo_pel_curr:{n_par_ele}')
    w_arr = sp.symbols('w:2')
    r_arr = sp.symbols('r:2')
    f_arr = sp.symbols('f:2')
    g_arr = sp.symbols('g:4')
    X_old, Y_old = sp.symbols('X_old Y_old')


    return (x, xi, h_e, beta_e, beta_curr, qe, a_arr, b_arr,
            c_arr, d_arr, e_arr, A_e, E_e, J_e, ro_e, T, fo_E, X_old, Y_old,
            Qglo_pel_curr, w_arr, r_arr, f_arr, g_arr)


def define_newton_equation(x: sp.Symbol, coeffs: List[sp.symbols]) -> sp.Expr:
    """
    Define a Newton polynomial equation.

    Args:
        x (sp.Symbol): The variable of the polynomial
        coeffs (List[sp.Symbol]): The coefficients of the polynomial

    Returns:
        sp.Expr: The final polynomial equation
    """
    equation = sum(c * x**i for i, c in enumerate(coeffs))
    return equation


def define_langrange_equation(xi: sp.Symbol, he: sp.Symbol, 
                              type_beam: Optional[int] = 1) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Define the Langrange shape functions for the beam and rod elements.
    This uses the Hermite cubic shape functions.

    Args:
        xi: The local coordinate
        he: The element length

    Returns:
        Tuple[List[sp.Expr], List[sp.Expr]]: The beam and rod shape functions
    """
    if type_beam == 1:
        N_beam_shape = sp.zeros(4, 1)  # Use sp.zeros to initialize a column matrix
        N_beam_shape[0] = 1 - 3*xi**2 + 2*xi**3
        N_beam_shape[1] = he * (xi - 2*xi**2 + xi**3)  # Corrected variable name to he
        N_beam_shape[2] = 3*xi**2 - 2*xi**3
        N_beam_shape[3] = he * (-xi**2 + xi**3)  # Corrected variable name to he
        return N_beam_shape
    
    N_rod_shape = sp.zeros(2, 1)
    N_rod_shape[0] = 1 - xi
    N_rod_shape[1] = xi

    return N_rod_shape


def compute_v_u(qe: List[sp.symbols], beta_e: sp.Symbol
                ) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute the local displacements v and u.

    Args:
        qe (List[sp.Symbol]): The local displacement vector
        beta_e (sp.Symbol): The angle of displacement

    Returns:
        Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]: The local displacements v and u
    """
    v1 = -qe[0] * sp.sin(beta_e) + qe[1] * sp.cos(beta_e)
    u1 = qe[0] * sp.cos(beta_e) + qe[1] * sp.sin(beta_e)
    v2 = -qe[3] * sp.sin(beta_e) + qe[4] * sp.cos(beta_e)
    u2 = qe[3] * sp.cos(beta_e) + qe[4] * sp.sin(beta_e)
    return v1, u1, v2, u2


def define_equilibrium_langrange(beam_type: str, u_beam: sp.Expr, v_beam: sp.Expr, alpha_beam: sp.Expr,
                                  xi: sp.Symbol, h_e: sp.Symbol, v1: sp.Expr, u1: sp.Expr, v2: sp.Expr,
                                  u2: sp.Expr, qe: List[sp.symbols]) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """
    Define the equilibrium equations for the beam or rod using Lagrange shape functions.

    Returns:
        List[sp.Expr]: The equilibrium equations for the beam or rod
    """
    N_beam_shape = define_langrange_equation(xi, h_e, type_beam=1)
    N_rod_shape = define_langrange_equation(xi, h_e, type_beam=0)
    theta_beam = sp.Expr(0)
    
    # Compute local displacements
    v_rod = N_rod_shape[0] * v1 + N_rod_shape[1] * v2
    u_rod = N_rod_shape[0] * u1 + N_rod_shape[1] * u2

    if beam_type == "bernoulli":
        v_beam = N_beam_shape[0]*v1 + N_beam_shape[1]*qe[2] + N_beam_shape[2]*v2 + N_beam_shape[3]*qe[5]
        u_beam = N_rod_shape[0]*u1 + N_rod_shape[1]*u2 - z * sp.diff(v_beam, xi)
    else:  # Timoshenko
        v_beam = N_beam_shape[0]*v1 + N_beam_shape[1]*qe[2] + N_beam_shape[2]*v2 + N_beam_shape[3]*qe[5]
        theta_beam = N_beam_shape[0]*qe[2] + N_beam_shape[2]*qe[5]
        u_beam = N_rod_shape[0]*u1 + N_rod_shape[0]*u2 - z * theta_beam

    return v_beam, u_beam, theta_beam, v_rod, u_rod,


def define_equilibrium_equations(beam_type: str, expressions: List[sp.Expr],
                                 x: sp.Symbol, h_e: sp.Symbol, v1: sp.Expr, u1: sp.Expr, v2: sp.Expr,
                                 u2: sp.Expr, qe: List[sp.symbols]) -> List[sp.Expr]:
    """
    Define the equilibrium equations for the beam or rod.

    Returns:
        List[sp.Expr]: The equilibrium equations for the beam or rod
    """
    if beam_type == "bernoulli":
        v_beam, u_beam = expressions[1], expressions[0]
        return [
            v_beam.subs(x, 0) - v1,
            sp.diff(v_beam, x).subs(x, 0) - qe[2],
            v_beam.subs(x, h_e) - v2,
            sp.diff(v_beam, x).subs(x, h_e) - qe[5],
            u_beam.subs(x, 0) - u1,
            u_beam.subs(x, h_e) - u2
        ]
    else:
        v_beam, u_beam, alpha_beam = expressions[1], expressions[0], expressions[2]
        return [
            u_beam.subs(x, 0) - u1,
            u_beam.subs(x, h_e) - u2,
            v_beam.subs(x, 0) - v1,
            v_beam.subs(x, h_e) - v2,
            alpha_beam.subs(x, 0) - qe[2],
            alpha_beam.subs(x, h_e) - qe[5]
        ]


def define_rod_equations(u_rod: sp.Expr, v_rod: sp.Expr, x: sp.Symbol, 
                         h_e: sp.Symbol, v1: sp.Expr, u1: sp.Expr, v2: sp.Expr,
                         u2: sp.Expr) -> List[sp.Expr]:
    """
    Define the equilibrium equations for the rod

    Returns:
        List[sp.Expr]: The equilibrium equations for the rod
    """
    return [
        v_rod.subs(x, 0) - v1,
        u_rod.subs(x, 0) - u1,
        v_rod.subs(x, h_e) - v2,
        u_rod.subs(x, h_e) - u2
    ]


def apply_boundary_conditions(K: np.array, M: np.array, W: np.array,
                              tol: float = 1e-5) -> Tuple[np.array, np.array]:
    """
    Applies the boundary conditions to the stiffness and mass matrices.

    Returns:
        Numpy arrays of the stiffness and mass matrices with the boundary conditions applied
    """
    indices = np.where(W == 1)[0]

    # take the max value from the diagonal
    max_k = np.max(np.abs(np.diag(K)))
    min_m = np.max(np.abs(np.diag(M)))
    max_freq_sqrd = np.sqrt(max_k / min_m)

    # Set rows and columns to zero
    K[indices, :] = 0
    K[:, indices] = 0
    M[indices, :] = 0
    M[:, indices] = 0

    # Correctly set diagonal elements for these indices
    for index in indices:
        K[index, index] = max_freq_sqrd / tol
        M[index, index] = max_freq_sqrd * tol

    return K, M



################## Analysis Functions ##################
def calculate_energies(beam_type, ve_beam, ue_beam, alpha_e_beam, ve_rod,
                       ue_rod, x, h_e, E_e, J_e, A_e, ro_e, G, k_shear):
    """
    Calculate the potential and kinetic energies of beams and rods.

    Returns:
        The potential and kinetic energies.
    """
    # Calculate chi_beam and eps_beam
    if beam_type == "bernoulli":
        chi_beam = sp.diff(sp.diff(ve_beam, x), x)
        eps_beam = sp.diff(ue_beam, x)
        pot_beam = 1 / 2 * sp.integrate(E_e * J_e * chi_beam**2 + E_e * A_e * eps_beam**2, (x, 0, h_e))
    else:
        eps_beam = sp.diff(ue_beam, x)
        gamma_beam = sp.diff(ve_beam, x) - alpha_e_beam
        chi_beam = sp.diff(alpha_e_beam, x)

        # Note that k_shear and G must be changed in case they are not constant
        pot_beam = 1 / 2 * sp.integrate(E_e * J_e * chi_beam**2 + E_e * A_e * eps_beam**2 + k_shear * G[0] * A_e * gamma_beam**2, (x, 0, h_e))
    
    kin_beam = 1 / 2 * ro_e * A_e * sp.integrate(ve_beam**2 + ue_beam**2, (x, 0, h_e))
    
    eps_rod = sp.diff(ue_rod, x)
    pot_rod = 1 / 2 * sp.integrate(E_e * A_e * eps_rod**2, (x, 0, h_e))
    kin_rod = 1 / 2 * ro_e * A_e * sp.integrate(ve_rod**2 + ue_rod**2, (x, 0, h_e))
    
    return pot_beam, kin_beam, pot_rod, kin_rod


def calculate_displacement_equations(x, xi, h_e, beta_e, qe, a_arr, b_arr, c_arr, d_arr, e_arr,
                                     beam_type, use_lagrangian: bool = True):
    """
    Calculate the beam displacement equations for beam (bernoulli or
    timoshenko) and rod.

    Returns:
        Displacement functions for the beam in the u and v directions
    """
    # Compute local displacements
    v1, u1, v2, u2 = compute_v_u(qe, beta_e)

    # Define beam displacement equations
    if use_lagrangian:
        v_beam, u_beam, theta_beam, v_rod, u_rod = define_equilibrium_langrange(beam_type, u_beam, v_beam, alpha_beam,
                                                                    xi, h_e, v1, u1, v2, u2, qe)
        # Lambdify ve_beam and ue_beam
        ve_beam_func = lambdify((xi, qe, h_e, beta_e), ve_beam, "numpy")
        ue_beam_func = lambdify((xi, qe, h_e, beta_e), ue_beam, "numpy")
        ve_rod_func = lambdify((xi, qe, h_e, beta_e), ve_rod, "numpy")
        ue_rod_func = lambdify((xi, qe, h_e, beta_e), ue_rod, "numpy")
    else: # Newton Interpolation
        if beam_type == "bernoulli":
            u_beam = define_newton_equation(x, a_arr)
            v_beam = define_newton_equation(x, c_arr)
            alpha_beam = sp.Expr(0)
        else:
            u_beam = define_newton_equation(x, a_arr)
            v_beam = define_newton_equation(x, b_arr)
            alpha_beam = define_newton_equation(x, e_arr)
        
        u_rod = define_newton_equation(x, b_arr)
        v_rod = define_newton_equation(x, d_arr)

        # Define equilibrium equations
        equations = define_equilibrium_equations(beam_type, [u_beam, v_beam, alpha_beam],
                                                x, h_e, v1, u1, v2, u2, qe)
        equations_rod = define_rod_equations(u_rod, v_rod, x, h_e, v1, u1, v2, u2)

        # Define equilibrium equations
        if beam_type == "bernoulli":
            sol = sp.solve(equations, a_arr + c_arr)
            alpha_e_beam = sp.Expr(0)
        else:
            sol = sp.solve(equations, a_arr + b_arr + e_arr)
            alpha_e_beam = alpha_beam.subs(sol)

        ve_beam = v_beam.subs(sol)
        ue_beam = u_beam.subs(sol)

        sol_rod = sp.solve(equations_rod, b_arr + d_arr)
        ve_rod = v_rod.subs(sol_rod)
        ue_rod = u_rod.subs(sol_rod)

        # Lambdify ve_beam and ue_beam
        ve_beam_func = lambdify((x, qe, h_e, beta_e), ve_beam, "numpy")
        ue_beam_func = lambdify((x, qe, h_e, beta_e), ue_beam, "numpy")

        ve_rod_func = lambdify((x, qe, h_e, beta_e), ve_rod, "numpy")
        ue_rod_func = lambdify((x, qe, h_e, beta_e), ue_rod, "numpy")

    return ve_beam_func, ue_beam_func, ve_beam, ue_beam, ve_rod_func, ue_rod_func, ve_rod, ue_rod, alpha_e_beam


def construct_lambdified_matrices(n_par_ele, pot_beam, kin_beam, pot_rod, kin_rod, qe, h_e, A_e, E_e, J_e, beta_e, ro_e):
    """
    Constructs and lambdifies the local K and M matrices.
    
    Returns:
        Lambdified functions for K and M matrices for beams and rods.
    """
    K_beam = sp.Matrix.zeros(n_par_ele)
    M_beam = sp.Matrix.zeros(n_par_ele)
    K_rod = sp.Matrix.zeros(n_par_ele)
    M_rod = sp.Matrix.zeros(n_par_ele)

    # Compute K_beam and M_beam
    for i in range(n_par_ele):
        for j in range(n_par_ele):
            K_beam[i, j] = sp.diff(sp.diff(pot_beam, qe[i]), qe[j])
            M_beam[i, j] = sp.diff(sp.diff(kin_beam, qe[i]), qe[j])
            K_rod[i, j] = sp.diff(sp.diff(pot_rod, qe[i]), qe[j])
            M_rod[i, j] = sp.diff(sp.diff(kin_rod, qe[i]), qe[j])

    # Create lambdified functions for K and M matrices
    K_beam_func = lambdify((h_e, A_e, E_e, J_e, beta_e), K_beam)
    M_beam_func = lambdify((h_e, A_e, E_e, J_e, beta_e, ro_e), M_beam)
    K_rod_func = lambdify((h_e, A_e, E_e, J_e, beta_e), K_rod)
    M_rod_func = lambdify((h_e, A_e, E_e, J_e, beta_e, ro_e), M_rod)

    return K_beam, M_beam, K_rod, M_rod, K_beam_func, M_beam_func, K_rod_func, M_rod_func


def assemble_global_matrices(n_par_ele: int, n_par_tot: int, n_ele_tot: int, K_beam_func: sp.lambdify,
                             M_beam_func: sp.lambdify, K_rod_func: sp.lambdify, M_rod_func: sp.lambdify,
                             h: sp.Symbol, A: sp.Symbol, E: sp.Symbol, J: sp.Symbol, beta: sp.Symbol,
                             ro: sp.Symbol, pel: List, n_rods: int) -> Tuple[np.array, np.array]:
    """
    Assembles the global stiffness and mass matrices using the lambdified functions for element matrices.

    Returns:
        Numeric arrays of the global stiffness and mass matrices.
    """
    # Initialize element stiffness matrix (Ke) and global stiffness matrix (K)
    K = np.zeros((n_par_tot, n_par_tot))
    M = np.zeros((n_par_tot, n_par_tot))

    # Pre-compute beam and rod indices
    beam_indices = np.arange(n_ele_tot - n_rods)
    rod_indices = np.arange(n_ele_tot - n_rods, n_ele_tot)
    logging.debug(f"Beam indices: {beam_indices} \nRod indices: {rod_indices}")
    
    # Process beams
    for e in beam_indices:
        Ke = K_beam_func(h[e], A[e], E[e], J[e], beta[e])
        Me = M_beam_func(h[e], A[e], E[e], J[e], beta[e], ro[e])
        idx = pel[e, :] - 1  # Adjust for 0-based indexing
        K[np.ix_(idx, idx)] += Ke
        M[np.ix_(idx, idx)] += Me

    # Process rods
    for e in rod_indices:
        Ke = K_rod_func(h[e], A[e], E[e], J[e], beta[e])
        Me = M_rod_func(h[e], A[e], E[e], J[e], beta[e], ro[e])
        idx = pel[e, :] - 1  # Adjust for 0-based indexing
        K[np.ix_(idx, idx)] += Ke
        M[np.ix_(idx, idx)] += Me

    # Convert K and M to NumPy arrays
    K = np.array(K).astype(np.float32)
    M = np.array(M).astype(np.float32)

    return K, M


def compute_eigenvalues_and_eigenvectors(K: np.array, M: np.array, method: str = 'numpy',
                                         filter_numerical_stability: bool = False,
                                         threshold: float = 1e-10) -> Tuple[np.array, np.array]:
    """
    Compute the eigenvalues (λ: w**2 natural frequencies)and 
    eigenvectors (ϕ: mode shape) of the stiffness and mass matrices.

    Args:
        K: Stiffness matrix
        M: Mass matrix
        method: 'scipy' for scipy.linalg.eigh or 'numpy' for np.linalg.eig
        filter_numerical_stability: Boolean to indicate if filtering should be applied
        threshold: Threshold for filtering small eigenvalues for numerical stability

    Returns:
        The real part of the eigenvalues (frequency) and the normalized eigenvectors (modes of vibration)
    """
    if method == 'numpy':
        lamb, phis = np.linalg.eig(np.linalg.inv(M) @ K)
    else:
        if method != 'scipy':
            print("Invalid method. Defaulting to scipy.")
        lamb, phis = scipy.linalg.eigh(K, M)


    # Get the indices that would sort lamb in descending order
    idx = np.argsort(lamb)[::-1]

    # Sort lamb and phis and take the real part
    lamb_r = np.real(lamb[idx])
    phis_r = np.real(phis[:, idx])

    if filter_numerical_stability:
        # Filter for numerical stability
        valid_indices = lamb_r > threshold
        lamb_r = lamb_r[valid_indices]
        phis_r = phis_r[:, valid_indices]

    # Normalize eigenvectors
    n_par_tot = len(lamb_r)
    phis_norm = np.zeros((phis_r.shape[0], n_par_tot))

    for i in range(n_par_tot):
        c = np.sqrt(np.dot(phis_r[:, i].T, M @ phis_r[:, i]))
        phis_norm[:, i] = phis_r[:, i] / c

    verification = np.array([np.dot(phis_norm[:, i].T, M @ phis_norm[:, i]) for i in range(n_par_tot)])
    if not np.allclose(verification, 1):
        logging.warning("Verification failed for eigenvectors. Results may be inaccurate.")
    logging.debug("Verification results for each eigenvector (should all be 1):\n%s", verification)

    verification_sum = np.sum(verification)
    if not np.isclose(verification_sum, n_par_tot):
        logging.warning("Sum of verification values is not equal to number of eigenvectors. Results may be inaccurate.")
    logging.debug("Sum of verification values (should be equal to number of eigenvectors): %s", verification_sum)

    return lamb_r, phis_norm


def get_mode_indices(lamb_r: np.array, phis_norm: np.array, 
                     n_plots: int) -> Tuple[np.array, np.array]:
    """
    Calculate the periods to get the top contributing index_modes.
    Periods are validated to be real numbers and positive.

    Args:
        lamb_r: Real part of the eigenvalues
        phis_norm: Normalized eigenvectors
        n_plots: Number of modes to plot

    Returns:
        Numpy array of the indices of the largest n_plots periods and sorted periods
    """
    # Calculate periods
    period = 2 * np.pi / np.sqrt(lamb_r)

    # Show how many invalid periods are there
    logging.debug("Number of invalid periods: %s", np.sum(~np.isfinite(period)))
    logging.debug("Number of negative periods: %s", np.sum(period < 0))

    # Ensure periods are all real numbers and valid
    valid_indices = np.isfinite(period) & (period > 0)
    period = period[valid_indices]
    phis_norm = phis_norm[:, valid_indices]

    # Sort periods for better representation
    sorted_period = np.sort(period)

    # Find the indices of the largest n_plots periods
    index_modes = np.argpartition(period, -n_plots)[-n_plots:]

    # Sort index_modes so that the modes are in descending order of period
    index_modes = index_modes[np.argsort(period[index_modes])][::-1]

    return index_modes, sorted_period, period


def calculate_global_displacements(Qglo_pel_curr, beta_e, h_e, x, xi, f_arr, 
                                    g_arr, w_arr, r_arr, beam_type,
                                    X_old, Y_old, use_lagrangian: bool = False) -> Tuple:
    """
    Calculate the global displacements by solving the local symbolic
    equilibrium equations.

    Returns:
        Lambda functions for the global displacements
    """

    _, _, v_beam, u_beam, _, _, v_rod, u_rod, _ = calculate_displacement_equations(x, xi, h_e, beta_e, Qglo_pel_curr, f_arr, r_arr, g_arr, w_arr, g_arr,
                                     beam_type, use_lagrangian)

    # Define global displacements
    u_glo_beam = u_beam * sp.cos(beta_e) - v_beam * sp.sin(beta_e)
    v_glo_beam = u_beam * sp.sin(beta_e) + v_beam * sp.cos(beta_e)
    u_glo_rod = u_rod * sp.cos(beta_e) - v_rod * sp.sin(beta_e)
    v_glo_rod = u_rod * sp.sin(beta_e) + v_rod * sp.cos(beta_e)
    

    # Define new coordinates
    if use_lagrangian:
        X_new_beam = X_old + xi * h_e * sp.cos(beta_e) + u_glo_beam
        Y_new_beam = Y_old + xi * h_e * sp.sin(beta_e) + v_glo_beam
        X_new_rod = X_old + xi * h_e * sp.cos(beta_e) + u_glo_rod
        Y_new_rod = Y_old + xi * h_e * sp.sin(beta_e) + v_glo_rod
    else:
        X_new_beam = X_old + x * sp.cos(beta_e) + u_glo_beam
        Y_new_beam = Y_old + x * sp.sin(beta_e) + v_glo_beam
        X_new_rod = X_old + x * sp.cos(beta_e) + u_glo_rod
        Y_new_rod = Y_old + x * sp.sin(beta_e) + v_glo_rod


    args = (X_old, Y_old, beta_e, h_e) + tuple(Qglo_pel_curr)
    if use_lagrangian:
        X_new_beam_func = sp.lambdify((xi,) + args, X_new_beam, "numpy")
        Y_new_beam_func = sp.lambdify((xi,) + args, Y_new_beam, "numpy")
        X_new_rod_func = sp.lambdify((xi,) + args, X_new_rod, "numpy")
        Y_new_rod_func = sp.lambdify((xi,) + args, Y_new_rod, "numpy")
    else:
        X_new_beam_func = sp.lambdify((x,) + args, X_new_beam, "numpy")
        Y_new_beam_func = sp.lambdify((x,) + args, Y_new_beam, "numpy")
        X_new_rod_func = sp.lambdify((x,) + args, X_new_rod, "numpy")
        Y_new_rod_func = sp.lambdify((x,) + args, Y_new_rod, "numpy")

    return X_new_beam, Y_new_beam, X_new_rod, Y_new_rod, X_new_beam_func, Y_new_beam_func, X_new_rod_func, Y_new_rod_func



################## Verifications functions ##################
def print_matrix(matrix: np.array, width: int = 9, precision: int = 1, row_labels: Optional[List[str]] = None,
                 col_labels: Optional[List[str]] = None) -> None:
    """
    Print a matrix in a more readable format.

    Args:
        matrix: The matrix to print
        width: The width of each column
        precision: The number of decimal places to show
        row_labels (Optional[List[str]], optional): Row labels. Defaults to numbering from 1 to n.
        col_labels (Optional[List[str]], optional): Column labels. Defaults to numbering from 1 to n.
    """
    if row_labels is None:
        row_labels = range(1, len(matrix) + 1)
    if col_labels is None:
        col_labels = range(1, len(matrix[0]) + 1)

    # Header row
    print(" " * width, end="")
    for label in col_labels:
        print(f"{label:>{width}}", end="")
    print()

    # Matrix rows
    for i, row in enumerate(matrix):
        print(f"{row_labels[i]:>{width}}", end="")
        for val in row:
            # Ensure the value is treated as a float for formatting
            try:
                formatted_val = f"{float(val):.{precision}e}"
            except ValueError:
                # If conversion to float fails, print as is
                formatted_val = str(val)
            print(f"{formatted_val:>{width}}", end="")
        print()


import numpy as np

def check_matrix(matrix: np.ndarray, atol: float = 1e-8) -> None:
    """
    Check if a matrix is symmetric, well-conditioned, positive definite, and diagonally dominant.
    
    Args:
        matrix: The matrix to check
        atol: The absolute tolerance for the condition check
    """
    # Symmetry check
    if np.allclose(matrix, matrix.T, atol=atol):
        print("Matrix is symmetric.")
    else:
        print("Matrix is not symmetric.")
    
    # Conditioning check
    try:
        cond_number = np.linalg.cond(matrix)
        if cond_number < 1 / atol:
            print(f"Matrix is well-conditioned (Condition number: {cond_number:.2e}).")
        else:
            print(f"Matrix is ill-conditioned (Condition number: {cond_number:.2e}).")
    except np.linalg.LinAlgError:
        print("Condition number could not be computed (possibly singular matrix).")
    
    # Positive definiteness check
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues < 0):
            print("Matrix is not positive definite.")
        else:
            print("Matrix is positive definite.")
    except np.linalg.LinAlgError:
        print("Eigenvalues could not be computed.")
    
    # Diagonal dominance check
    row_sums = np.sum(np.abs(matrix), axis=1) - np.abs(np.diag(matrix))
    diagonal_elements = np.abs(np.diag(matrix))
    is_dominant = diagonal_elements >= row_sums
    num_wrong = np.size(matrix, 0) - np.sum(is_dominant)
    
    if num_wrong == 0:
        print("Matrix is diagonally dominant.")
    else:
        print(f"Number of rows not diagonally dominant: {num_wrong}")
