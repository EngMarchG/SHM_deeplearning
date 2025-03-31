import numpy as np
from typing import List, Tuple


def nodal_coords(n_nod_tot: int, n_dim: int, n_columns: int, spacing: float, 
                 height: float, n_bot_beams: int, truss_mode: str) -> np.ndarray:
    """
    Calculate the coordinate of the nodes in the assembly.

    Args:
        n_nod_tot (int): Total number of nodes.
        n_dim (int): Number of dimensions.
        n_columns (int): Number of columns.
        spacing (float): Spacing between columns (length of beams)
        height (float): Height of the columns .
        n_bot_beams (int): Number of bottom beams.
        truss_mode (str): Mode of the truss ("warren" or other)
                        currently supports pratt and howe as other.

    Returns:
        np.ndarray: The nodal coordinates of the assembly.
    """
    nodal_coord = np.zeros((n_nod_tot, n_dim), dtype=float)

    # Calculate the nodal coordinates
    if truss_mode == "simple" or truss_mode == "simple_cant":
        for i in range(n_nod_tot):
            nodal_coord[i] += [i * spacing, 0]
    elif truss_mode != "warren":
        for i in range(n_columns + 2):
            nodal_coord[i] += [i * spacing, 0]
            if 0 < i < n_columns + 1:
                nodal_coord[i + n_columns + 1] += [i * spacing, height]
    else:
        for i in range(n_nod_tot - n_bot_beams):
            nodal_coord[i] += [i * spacing, 0]
            if i > 0:
                nodal_coord[i + n_bot_beams] += [i * spacing - spacing / 2, height]

    return nodal_coord


def pel_ele(par: np.ndarray, n_columns: int, n_beams: int, n_rods: int, 
            n_par_nod: int, n_nod_tot: int, n_ele_tot: int, n_bot_beams: int, 
            truss_mode: str, skip_rod: List[int] = []) -> np.ndarray:
    """
    Calculate the parameter-element numbering relation.
    Built from Beam -> Column -> Rods (considering skipped rods).

    Args:
        par (np.ndarray): Parameters for the elements.
        n_columns (int): Number of columns.
        n_beams (int): Number of beams.
        n_rods (int): Number of rods.
        n_par_nod (int): Number of parameters per node.
        n_nod_tot (int): Total number of nodes.
        n_ele_tot (int): Total number of elements.
        n_bot_beams (int): Number of bottom beams.
        truss_mode (str): Mode of the truss ("warren", "pratt", etc.).
        skip_rod (List[int]): Rods to skip. 0-indexed.

    Returns:
        np.ndarray: The parameter-element numbering relation.
    """
    if "simple" in truss_mode:
        pel = beam_pars(par, n_beams, n_par_nod, n_nod_tot, n_bot_beams, truss_mode)
    else:
        pel = np.zeros((n_ele_tot - len(skip_rod), 2 * n_par_nod), dtype=int)

        # Calculate the element parameter relations
        pel[:n_beams] = beam_pars(par, n_beams, n_par_nod, n_nod_tot, n_bot_beams, truss_mode)
        if truss_mode != "warren":
            pel[n_beams:n_beams + n_columns] = column_pars(par, n_columns, n_par_nod)
        pel[n_beams + n_columns:] = rod_pars(par, n_rods, n_par_nod, n_nod_tot, n_bot_beams, truss_mode, skip_rod)
    
    return pel


def beam_pars(par: np.ndarray, n_beams: int, n_par_nod: int,
              n_nod_tot: int, n_bot_beams: int,
              truss_mode: str) -> np.ndarray:
    """
    Calculate the relevant beam DoFs.

    Args:
        par (np.ndarray): Parameters for the elements.
        n_beams (int): Number of beams.
        n_par_nod (int): Number of parameters per node.
        n_nod_tot (int): Total number of nodes.
        n_bot_beams (int): Number of bottom beams.

    Returns:
        np.ndarray: The beam DoFs.
    """
    beams = np.zeros((n_beams, 2 * n_par_nod))

    if "simple" in truss_mode:
        for i in range(n_beams):
            beams[i, :n_par_nod] = par[i]
            beams[i, n_par_nod:] = par[i + 1]
    else:
        beams[:n_bot_beams, :n_par_nod] = par[:n_bot_beams]
        beams[:n_bot_beams, n_par_nod:] = par[1:n_bot_beams + 1]

        beams[n_bot_beams:, :n_par_nod] = par[n_bot_beams + 1:n_nod_tot - 1]
        beams[n_bot_beams:, n_par_nod:] = par[n_bot_beams + 2:]

    return beams


def column_pars(par: np.ndarray, n_columns: int, n_par_nod: int) -> np.ndarray:
    """
    Calculate the relevant column DoFs.

    Args:
        par (np.ndarray): Parameters for the elements.
        n_columns (int): Number of columns.
        n_par_nod (int): Number of parameters per node.

    Returns:
        np.ndarray: The column DoFs.
    """
    columns = np.zeros((n_columns, 2 * n_par_nod))
    for i in range(1, n_columns + 1):
        columns[i - 1][:n_par_nod] = par[i]
        columns[i - 1][n_par_nod:] = par[i + n_columns + 1]

    return columns


def rod_pars(par: np.ndarray, n_rods: int, n_par_nod: int, n_nod_tot: int, 
             n_bot_beams: int, truss_mode: str, skip_rod: List[int] = []) -> np.ndarray:
    """
    Calculate the relevant rod DoFs.

    Args:
        par (np.ndarray): Parameters for the elements.
        n_rods (int): Number of rods.
        n_par_nod (int): Number of parameters per node.
        n_nod_tot (int): Total number of nodes.
        n_bot_beams (int): Number of bottom beams.
        truss_mode (str): Mode of the truss ("warren" or other).
        skip_rod (List[int]): The rods to skip. From left to right.

    Returns:
        np.ndarray: The rod DoFs.
    """
    rods = np.zeros((n_rods - len(skip_rod), 2 * n_par_nod))
    
    count = 0
    skips = 0

    for i in range(n_nod_tot - n_bot_beams - 1):
        for j in range(2):
            if count not in skip_rod:
                rods[count - skips][n_par_nod:] = par[i + n_bot_beams + 1]
                if j > 0 and truss_mode != "warren":
                    rods[count - skips][:n_par_nod] = par[2 * j + i]
                else:
                    rods[count - skips][:n_par_nod] = par[j + i]
            else:
                skips += 1
            count += 1

    return rods


def fill_ele_nod(n_ele_tot: int, n_par_nod: int, pel: np.ndarray, 
                 skip_rod: List[int] = []) -> np.ndarray:
    """
    Fill the element-nodal matrix.

    Args:
        n_ele_tot (int): Total number of elements.
        n_par_nod (int): Number of parameters per node.
        pel (np.ndarray): Parameter-element numbering relation.
        skip_rod (List[int]): Rods to skip.

    Returns:
        np.ndarray: The element-nodal matrix.
    """
    ele_nod = np.zeros((n_ele_tot - len(skip_rod), 2), dtype=int)

    ele_nod[:, 0] = pel[:, 0] // n_par_nod
    ele_nod[:, 1] = pel[:, n_par_nod] // n_par_nod

    return ele_nod


# If main is called, run the test
if __name__ == "__main__":
    n_par_nod = 3
    par = np.arange(1, 7*n_par_nod+1).reshape(7, n_par_nod)
    n_beams = 5
    truss_mode = "warren"
    n_bot_beams = 3
    n_nod_tot = 7
    n_rods = 6
    n_dim = 3
    n_columns = 2

    if truss_mode == "warren":
        n_columns = 0
    n_ele_tot = n_beams + n_columns + n_rods

    print(par)
    # beams =  beam_pars(par, n_beams, n_par_nod, n_nod_tot, n_bot_beams)
    # rods = rod_pars(par, n_rods, n_par_nod, n_bot_beams, skip_rod=[])
    pel = pel_ele(par, n_columns, n_beams, n_rods, n_par_nod, n_nod_tot, n_ele_tot, n_bot_beams, truss_mode, skip_rod)

    print(pel)