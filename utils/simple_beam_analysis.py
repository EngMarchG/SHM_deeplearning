import numpy as np
from typing import List, Optional

def define_forces(W: np.ndarray, node_idxs: List, force_list: List, 
                  n_par_nod: int, node_pos: Optional[List] = None):
    """
    Define the forces acting on the beam. Index is 0-based.
    """
    ext_forces = np.zeros((W.shape[0]), dtype=np.float32)

    if len(node_idxs) != len(force_list) // n_par_nod:
        raise ValueError("# of forces/n_par_nod != # nodes.")
    if node_pos and len(node_pos) != len(force_list) // n_par_nod:
        raise ValueError("# of forces/n_par_nod != # nodes positions.")

    if not node_pos:
        for i, idx in enumerate(node_idxs):
            ext_idx = idx * n_par_nod
            force_idx = i * n_par_nod
            ext_forces[ext_idx:ext_idx + n_par_nod] = force_list[force_idx:force_idx + n_par_nod]
    else:
        for i, pos in enumerate(node_pos):
            ext_idx = pos * n_par_nod
            force_idx = i * n_par_nod
            ext_forces[ext_idx:ext_idx + n_par_nod] = force_list[force_idx:force_idx + n_par_nod]
    
    # Apply boundary conditions by making all the places where there is 1 in W to be 0 in ext_forces
    ext_forces = ext_forces * (1 - W)

    return ext_forces


################## Direct Stiffness Method for 2D Bernoulli Beam ##################
def direct_assemble_global_matrices(n_par_ele: int, n_par_tot: int, n_ele_tot: int,
                                    ele_nod: np.ndarray, h: np.ndarray,
                                    E: np.ndarray, J: np.ndarray, A: np.ndarray,
                                    beta: np.ndarray, pel: np.ndarray) -> np.ndarray:
    """
    Assemble the global stiffness matrix for 2D Bernoulli beam elements dynamically.

    Args:
        n_par_ele: Number of parameters per element (6 for 2 nodes with 3 DOFs each)
        n_par_tot: Total number of DOFs in the system
        n_ele_tot: Total number of elements
        ele_nod: Element-node connectivity array (n_ele_tot x 2)
        h: Element lengths array
        E: Young's modulus array
        J: Second moment of inertia array
        A: Cross-sectional area array
        beta: Element angles array (in radians)
        pel: DOF mapping array (n_ele_tot x n_par_ele)

    Returns:
        K: Global stiffness matrix of size (n_par_tot x n_par_tot)
    """
    K = np.zeros((n_par_tot, n_par_tot), dtype=np.float64)

    for e in range(n_ele_tot):
        L = h[e]
        EA = E[e] * A[e]
        EI = E[e] * J[e]
        
        # Local stiffness matrix for a Bernoulli beam element (6x6)
        EA_L = EA / L
        EI_L3 = EI / L**3
        EI_L2 = EI / L**2
        EI_L1 = EI / L

        # Assemble the local stiffness matrix in the local coordinate system
        k_local = np.array([
            [ EA_L,        0,         0,      -EA_L,       0,         0],
            [  0,     12*EI_L3,   6*EI_L2,      0,  -12*EI_L3,   6*EI_L2],
            [  0,      6*EI_L2,   4*EI_L1,      0,   -6*EI_L2,   2*EI_L1],
            [ -EA_L,       0,         0,       EA_L,       0,         0],
            [  0,   -12*EI_L3,  -6*EI_L2,      0,   12*EI_L3,  -6*EI_L2],
            [  0,      6*EI_L2,   2*EI_L1,      0,   -6*EI_L2,   4*EI_L1]
        ])
        
        # Transformation matrix for element orientation (6x6)
        c = np.cos(beta[e])
        s = np.sin(beta[e])
        T = np.array([
            [ c,   s,  0,   0,   0,  0],
            [-s,   c,  0,   0,   0,  0],
            [ 0,   0,  1,   0,   0,  0],
            [ 0,   0,  0,   c,   s,  0],
            [ 0,   0,  0,  -s,   c,  0],
            [ 0,   0,  0,   0,   0,  1]
        ])
        
        # Rotate local stiffness matrix to global coordinates
        k_global = T.T @ k_local @ T
        
        # Get global DOF indices
        idx = pel[e, :] - 1 # Global DOF indices for element 'e'
        
        # Assemble into global stiffness matrix
        K[np.ix_(idx, idx)] += k_global

    return K

def direct_apply_boundary_conditions(K: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Apply boundary conditions to global stiffness matrix using W array.
    W[i] = 1 means DOF i is restrained, W[i] = 0 means DOF i is free.
    
    Args:
        K: Global stiffness matrix
        W: Boundary condition array (1 for restrained, 0 for free DOFs)
    
    Returns:
        K: Modified stiffness matrix with boundary conditions applied
    """
    # Make copy to avoid modifying original
    K_bc = K.copy()
    max_k = np.max(np.abs(np.diag(K))) # Double check this
    
    # Find restrained DOFs
    restrained_dofs = np.where(W == 1)[0]
    
    # For each restrained DOF
    for dof in restrained_dofs:
        # Zero out row and column
        K_bc[dof, :] = 0
        K_bc[:, dof] = 0
        K_bc[dof, dof] = max_k

    return K_bc