import logging
import numpy as np
from typing import List, Optional, Tuple, Union

# Get the logger
logger = logging.getLogger(__name__)

def calculate_simple_essential_elements(span: float, spacing: Optional[float] = 0,
                                        truss_mode: str = "simple", beam_partition: int = 1,
                                        col_placements: Optional[List[float]] = None,
                                        skip_col: Optional[List[int]] = None,
                                        cantilever_sides: int = 1,
                                        add_extra_pt: bool = False) -> Tuple[int, int, int, int, int, int]:
    """
    Calculate the number of columns, nodes, rods, beams, and total elements for a simple beam.

    Args:
        span (float): The total length of the beam.
        spacing (float or List[float]): Distance between nodes or list of node positions.
        truss_mode (str): The type of beam ("simple", "simple_cant").
        beam_partition (int): Number of divisions in the beam.
        col_placements (List[float], optional): Custom positions of the columns (supports).
        skip_col (List[int], optional): Indices of columns to skip.

    Returns:
        Tuple[int, int, int, int, int, int]: Number of columns, total nodes, rods, beams, total elements, and bottom beams.
    """
    if not spacing:
        print("Spacing was not provided. Defaulting to 1. Hence a simply supported beam.")
        spacing = 1.0
    assert span % spacing == 0, "Spacing does not divide the span evenly."

    
    n_columns = int(span // spacing) + 1
    n_rods = 0  # Simple beams have no rods
    n_beams = (n_columns - 1) * beam_partition
    if truss_mode != "simple":
        if cantilever_sides > 2:
            print("Cantilever sides cannot be more than 2. Defaulting to 2.")
            cantilever_sides = 2
        n_beams += cantilever_sides * beam_partition
    
    n_nod_tot = n_beams + 1  # Number of nodes is one more than the number of beams

    if add_extra_pt and span/2 == n_nod_tot/2 * spacing:
        print("Extra point cannot be added at the center. Defaulting to False.")
        add_extra_pt = False

    if skip_col:
        n_nod_tot -= len(skip_col)

    if add_extra_pt:
        n_nod_tot += 1
        n_beams += 1

    n_ele_tot = n_beams  # Only beams are considered elements here
    n_bot_beams = 0  # Not applicable for simple beams

    return n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams


def calculate_simple_element_node(span: float, spacing: Union[float, List[float]], n_dim: int,
                                  n_par_nod: int, truss_mode: str = "simple",
                                  beam_partition: int = 1,
                                  col_placements: Optional[List[float]] = None,
                                  skip_col: Optional[List[int]] = None,
                                  cantilever_sides: int = 1,
                                  add_extra_pt: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate nodal coordinates, nodal-param relation, and element-node relationships for a simple beam.

    Args:
        span (float): The total length of the beam.
        spacing (float or List[float]): Distance between nodes or list of node positions.
        n_dim (int): Number of dimensions (usually 2).
        n_par_nod (int): Number of parameters per node (e.g., 2 for x and y coordinates).
        truss_mode (str): The type of beam ("simple", "simple_cant").
        skip_rod (List[int], optional): Rods to skip (not applicable for simple beams).
        beam_partition (int): Number of divisions in the beam.
        col_placements (List[float], optional): Custom positions of the columns (supports).
        skip_col (List[int], optional): Indices of columns to skip.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: Nodal coordinates, parameter matrix, parameter-element relation, element-node relation, and total parameters.
    """
    # Get essential elements
    n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams = calculate_simple_essential_elements(
        span, spacing, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides, add_extra_pt)

    n_par_tot = n_nod_tot * n_par_nod
    
    # Generate nodal coordinates
    nodal_coord = nodal_coords_simple(n_nod_tot, n_dim, spacing, span, n_beams,
                                      col_placements, skip_col, beam_partition,
                                      truss_mode, cantilever_sides, add_extra_pt)

    # Generate nodal parameters
    par = np.arange(1, n_par_tot + 1).reshape(n_nod_tot, n_par_nod)

    # Generate element-parameter relationships
    pel = pel_ele_simple(par, n_beams, n_par_nod)

    # Generate element-node relationships
    ele_nod = fill_ele_nod_simple(n_beams, n_par_nod, pel)

    return nodal_coord, par, pel, ele_nod, n_par_tot


def nodal_coords_simple(n_nod_tot: int, n_dim: int, spacing: Union[float, List[float]],
                        span: float, n_beams: int,
                        col_placements: Optional[List[float]] = None,
                        skip_col: Optional[List[int]] = None, beam_partition: int = 1,
                        truss_mode: str = "simple", cantilever_sides: int = 2,
                        add_extra_pt: bool = False) -> np.ndarray:
    """
    Generate nodal coordinates for a simple beam.

    Args:
        n_nod_tot (int): Total number of nodes.
        n_dim (int): Number of dimensions.
        spacing (float or List[float]): Spacing between nodes or list of node positions.
        span (float): Total length of the beam.
        col_placements (List[float], optional): Custom positions of the columns (supports).
        skip_col (List[int], optional): Indices of columns to skip.
        beam_partition (int): Number of divisions in the beam.
        truss_mode (str): The type of beam ("simple", "simple_cant").
        cantilever_sides (int): Number of cantilever sides. It will default to left side then right. 

    Returns:
        np.ndarray: The nodal coordinates.
    """
    if cantilever_sides > 2:
        print("Cantilever sides cannot be more than 2. Defaulting to 2.")
        cantilever_sides = 2
    
    if col_placements:
        node_positions = col_placements
    else:
        if isinstance(spacing, list):
            # Generate node positions from spacing list
            node_positions = [0]
            for s in spacing:
                node_positions.append(node_positions[-1] + s)
        else:
            # Evenly spaced nodes
            if truss_mode != "simple":
                # extra_span = (n_beams - cantilever_sides * beam_partition) * spacing
                # extra_nodes = (cantilever_sides * beam_partition) - (cantilever_sides - 1)
                if not add_extra_pt:
                    node_positions = [i * (span + span * cantilever_sides) / n_beams for i in range(n_nod_tot)]
                else:
                    node_positions = [i * (span + span * cantilever_sides) / n_beams for i in range(n_nod_tot-1)]
                    node_positions = node_positions[:n_nod_tot//2] + [span + span * cantilever_sides] + node_positions[n_nod_tot//2:]

            else:
                if not add_extra_pt:
                    node_positions = [i * span / n_beams for i in range(n_nod_tot)]
                else:
                    node_positions = [i * span / n_beams for i in range(n_nod_tot-1)]
                    node_positions = node_positions[:n_nod_tot//2] + [span] + node_positions[n_nod_tot//2:]
    
    # Apply skip_col if provided
    if skip_col:
        node_positions = [pos for idx, pos in enumerate(node_positions) if idx not in skip_col]
    
    nodal_coord = np.zeros((n_nod_tot, n_dim))
    
    logging.debug("In the nodal_coord_simple function we have: n_nod_tot: %d, \n\
                  node_positions: %s, n_beams: %d", n_nod_tot, node_positions, n_beams)
    for i in range(n_nod_tot):
        nodal_coord[i, 0] = node_positions[i]
        nodal_coord[i, 1] = 0 

    return nodal_coord


def pel_ele_simple(par: np.ndarray, n_beams: int, n_par_nod: int) -> np.ndarray:
    """
    Generate the parameter-element numbering relation for a simple beam with 2 nodes

    Args:
        par (np.ndarray): Parameter matrix.
        n_beams (int): Number of beams (elements).
        n_par_nod (int): Number of parameters per node.

    Returns:
        np.ndarray: The parameter-element numbering relation.
    """
    pel = np.zeros((n_beams, 2 * n_par_nod), dtype=int)
    for i in range(n_beams):
        pel[i, :n_par_nod] = par[i]
        pel[i, n_par_nod:] = par[i + 1]

    return pel


def fill_ele_nod_simple(n_beams: int, n_par_nod: int, pel: np.ndarray) -> np.ndarray:
    """
    Generate the element-node relationships for a simple beam.

    Args:
        n_beams (int): Number of beams (elements).
        n_par_nod (int): Number of parameters per node.
        pel (np.ndarray): Parameter-element numbering relation.

    Returns:
        np.ndarray: The element-node relationships.
    """
    ele_nod = np.zeros((n_beams, 2), dtype=int)
    # print(ele_nod, pel, n_beams)
    logging.debug("In the fill_ele_nod_simple function we have: n_beams: %d, \n\
                  ele_nod: %s, pel: %s", n_beams, ele_nod, pel)
    for i in range(n_beams):
        ele_nod[i, 0] = (pel[i, 0] - 1) // n_par_nod
        ele_nod[i, 1] = (pel[i, n_par_nod] - 1) // n_par_nod

    return ele_nod


def boundary_conditions_simple(spacing: int, n_par_nod: int, n_nod_tot: int, n_columns: int, col_placements: List,
                               cantilever_sides: int, beam_partition: int, truss_mode: str = "simple",
                               default_support: str = "roller", supports: List = ["pin", "roller"]) -> np.ndarray:
    """
    Generate boundary conditions for a simple beam or cantilever.
    If custom column placements are provided, it uses those; otherwise, 
    it defaults to evenly spaced supports. The function ensures that there are enough supports
    by defaulting to the specified default support type if necessary

    Args:
        spacing (int): Distance between nodes.
        n_par_nod (int): Number of parameters per node (e.g., 2 for x and y coordinates).
        n_nod_tot (int): Total number of nodes.
        n_columns (int): Number of columns (supports).
        col_placements (List): Custom positions of the columns (supports).
        beam_partition (int): Number of divisions in the beam.
        truss_mode (str, optional): The type of beam ("simple", "simple_cant"). Defaults to "simple".
        default_support (str, optional): Default support type if not enough supports are provided. Defaults to "roller".
        supports (List, optional): List of support types (e.g., ["pin", "roller"]). Defaults to ["pin", "roller"].

    Returns:
        np.ndarray: Array representing the boundary conditions for each node.
    """
    # Initialize the boundary conditions
    def support_dof(support, n_par_nod):
        temp = np.zeros(n_par_nod, dtype=int)
        if support == "roller":
            temp[1] = 1
        elif support == "pin":
            temp[:2] = 1
        else:
            temp[:] = 1
        return temp
    
    # TODO: Implement 
    adj_nod = 0 # Adjust the starting node
    adj_nod_end = 0 # Adjust the ending node
    adj_mode = 1 if truss_mode == "simple" else 0
    if truss_mode != "simple":
        adj_nod = beam_partition
        if cantilever_sides >= 2:
            adj_nod_end = beam_partition - 1
    
    if col_placements:
        if len(supports) < len(col_placements):
            print("Not enough supports provided. Defaulting to pin and roller.")
            supports = [supports[0]]
            supports.extend(default_support * (len(col_placements) - 1))
    elif len(supports) < n_columns:
        print(f"Not enough supports provided for boundary conditions. \
              Defaulting to pin then {default_support} for {n_columns - 1} columns.")
        supports = [supports[0]]
        supports.extend([default_support] * (n_columns - 1))
    
    temp = np.zeros((n_nod_tot * n_par_nod), dtype=int)
    counter = 0
    # Counter for incrementing supports
    # Range starts from the first non-cantiliver node till the last non-cantilever node
    # Movevment is based on the original beam subtracted by the number of columns
    try:
        for i in range(adj_nod, n_nod_tot - adj_nod_end, max(n_nod_tot - adj_nod - adj_nod_end - len(supports) + adj_mode, 1)):
            temp[i * n_par_nod:(i + 1) * n_par_nod] = support_dof(supports[counter], n_par_nod)
            counter += 1
    except IndexError:
        logging.error(f"Check the last non-cantilever node or support indexes \n\
                      adj_nod: {adj_nod}, n_nod_tot: {n_nod_tot}, adj_nod_end: {adj_nod_end}, n_columns: {n_columns} supports: {supports}")
    
    return temp


def insert_simple_node(node_pos: float, x_pos: np.ndarray, n_par_nod: int,
                       length_arr: np.ndarray, same_property_list: List[np.ndarray],
                       node_list: List[np.ndarray], W: np.ndarray, n_ele_tot: int,
                       n_par_tot: int, n_nod_tot: int) -> Tuple:
    """
    Insert a new node into the existing structure.
    This function updates the node positions, lengths, and relationships accordingly.

    Returns:
        Updated node positions, lengths, relationships, and counts.
    """
    if node_pos in x_pos:
        print("Attempted node insertion. Node already exists")
        return (x_pos, length_arr, *same_property_list, *node_list, W, n_ele_tot, n_par_tot, n_nod_tot)

    n_ele_tot += 1
    n_nod_tot += 1
    n_par_tot += n_par_nod
    node_idx = np.searchsorted(x_pos, node_pos)
    x_pos = np.insert(x_pos, node_idx, node_pos)
    W = np.insert(W, node_idx * n_par_nod, np.zeros(n_par_nod, dtype=np.int32), axis=0)

    length_arr = np.insert(length_arr, node_idx, x_pos[node_idx+1] - node_pos)
    length_arr[node_idx - 1] = node_pos - x_pos[node_idx - 1]

    def insert_node(node_idx, node_arr, n_par_nod, increment: int = 0):
        if not increment:
            node_arr = np.insert(node_arr, node_idx, node_arr[node_idx - 1], axis=0)
        else:
            node_arr = np.append(node_arr, [node_arr[-1] + np.arange(1, len(node_arr[-1]) + 1)], axis=0)
        return node_arr
    
    for i in range(len(same_property_list)):
        same_property_list[i] = insert_node(node_idx, same_property_list[i], n_par_nod)

    for i in range(len(node_list)):
        node_list[i] = insert_node(node_idx, node_list[i], n_par_nod, increment=1)
    
    # Fix pel
    last_pel = node_list[0][-2][n_par_nod:]  
    new_pel = last_pel + n_par_nod  
    node_list[0][-1] = np.concatenate((last_pel, new_pel))

    return (x_pos, length_arr, *same_property_list, *node_list, W, n_ele_tot, n_par_tot, n_nod_tot)


# Run some test cases if its the main module
if __name__ == "__main__":
    # Test case 1 with beam partition
    test_case = 1
    span = 25
    spacing = 12.5
    n_dim = 2
    n_par_nod = 3
    truss_mode = "simple"
    beam_partition = 2
    col_placements = None
    skip_col = None
    cantilever_sides = 2

    n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams = calculate_simple_essential_elements(
        span, spacing, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides)
    print(
        f"There are {n_columns} columns, {n_nod_tot} total nodes, {n_beams} beams and {n_ele_tot} total elements (excluding columns)"
    )

    nodal_coord, par, pel, ele_nod, n_par_tot = calculate_simple_element_node(
        span, spacing, n_dim, n_par_nod, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides)
    print("Nodal Coordinates:\n", nodal_coord)
    print("Parameter Matrix:\n", par)
    print("Parameter-Element Relationship:\n", pel)
    print("Element-Node Relationship:\n", ele_nod)
    print("Total Parameters:", n_par_tot)

    print(f"Test {test_case} test done successfully\n\n")

    # Test case 2 without beam partition
    test_case = 2
    span = 25
    spacing = 25
    n_dim = 2
    n_par_nod = 2
    truss_mode = "simple"
    beam_partition = 1
    col_placements = None
    skip_col = None

    n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams = calculate_simple_essential_elements(
        span, spacing, truss_mode, beam_partition, col_placements, skip_col)
    print(
        f"There are {n_columns} columns, {n_nod_tot} total nodes, {n_beams} beams and {n_ele_tot} total elements (excluding columns)"
    )

    nodal_coord, par, pel, ele_nod, n_par_tot = calculate_simple_element_node(
        span, spacing, n_dim, n_par_nod, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides)
    print("Nodal Coordinates:\n", nodal_coord)
    print("Parameter Matrix:\n", par)
    print("Parameter-Element Relationship:\n", pel)
    print("Element-Node Relationship:\n", ele_nod)
    print("Total Parameters:", n_par_tot)

    print(f"Test {test_case} test done successfully\n\n")


    # Test case 3 without beam partition simple cant
    test_case = 3
    span = 25
    spacing = 25
    n_dim = 2
    n_par_nod = 2
    truss_mode = "simple_cant"
    beam_partition = 1
    col_placements = None
    skip_col = None
    cantilever_sides = 2

    n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams = calculate_simple_essential_elements(
        span, spacing, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides)
    print(
        f"There are {n_columns} columns, {n_nod_tot} total nodes, {n_beams} beams and {n_ele_tot} total elements (excluding columns)"
    )

    nodal_coord, par, pel, ele_nod, n_par_tot = calculate_simple_element_node(
        span, spacing, n_dim, n_par_nod, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides)
    print("Nodal Coordinates:\n", nodal_coord)
    print("Parameter Matrix:\n", par)
    print("Parameter-Element Relationship:\n", pel)
    print("Element-Node Relationship:\n", ele_nod)
    print("Total Parameters:", n_par_tot)

    print(f"Test {test_case} test done successfully\n\n")


    # Test case 4 with beam partition simple cant
    test_case = 3
    span = 25
    spacing = 25
    n_dim = 2
    n_par_nod = 2
    truss_mode = "simple_cant"
    beam_partition = 2
    col_placements = None
    skip_col = None
    cantilever_sides = 1

    n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams = calculate_simple_essential_elements(
        span, spacing, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides)
    print(
        f"There are {n_columns} columns, {n_nod_tot} total nodes, {n_beams} beams and {n_ele_tot} total elements (excluding columns)"
    )

    nodal_coord, par, pel, ele_nod, n_par_tot = calculate_simple_element_node(
        span, spacing, n_dim, n_par_nod, truss_mode, beam_partition, col_placements, skip_col, cantilever_sides)
    print("Nodal Coordinates:\n", nodal_coord)
    print("Parameter Matrix:\n", par)
    print("Parameter-Element Relationship:\n", pel)
    print("Element-Node Relationship:\n", ele_nod)
    print("Total Parameters:", n_par_tot)

    print(f"Test {test_case} test done successfully\n\n")