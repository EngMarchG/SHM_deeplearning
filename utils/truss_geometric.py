import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from utils.truss_element_assembly import nodal_coords, pel_ele, fill_ele_nod

MIN_ANGLE = np.pi / 6
MAX_ANGLE = np.pi / 3

# Get the logger
logger = logging.getLogger(__name__)


################## Geometric Functions ##################
def calculate_max_height(span: float, angle: float, spacing: float = 0,
                         tol_round: Optional[List] = None) -> Tuple[float, float]:
    """
    Calculate the maximum height and spacing for the given span and angle.
    If spacing is given, no span/spacing ratio is respected.

    Args:
        span: The total length of the bridge.
        angle: The angle of the diagonal rods in radians.
        spacing: Pre-defined spacing between columns. If 0, it will be calculated.
        allowed_round: The number of decimal places to round the height to. first value is for height, second for spacing.
                    More angles will be found with higher values.

    Returns:
        A tuple containing the height and spacing if valid, otherwise (-1, 0).
    """
    if tol_round is None:
        tol_round = [3, 1]

    if not MIN_ANGLE <= angle <= MAX_ANGLE:
        logging.warning("The angle should be between %d and %d. "
                        "Defaulting to 45 degrees.", np.degrees(MIN_ANGLE), np.degrees(MAX_ANGLE))
        angle = np.radians(45)
    
    if spacing:
        height = round(spacing * np.tan(angle), tol_round[0])
        return height, spacing

    for i in range(15, 21):
        height = round(span / i, tol_round[0])
        spacing = round(height / np.tan(angle), tol_round[1])
        if span % spacing == 0: 
            return height, spacing

    return -1, 0


def try_angles(span: float, base_angle: int, lower_limit: float = 30, upper_limit: float = 60,
               tol_round: Optional[List] = None) -> Tuple[float, float, float]:
    """
    Try to find a valid height and spacing by adjusting the angle.
    
    Args:
        span: The total length of the bridge.
        base_angle: The base angle of the diagonal rods in degrees.

    Returns:
        A tuple containing the height, spacing, and adjusted angle in degrees if found, otherwise (-1, -1, -1).
    """
    if tol_round is None:
        tol_round = [3, 1]
        
    # Calculate the distance of the angle from pi/3 and pi/6 to use in range
    angle_diff = int(max(abs(base_angle - lower_limit), abs(base_angle - upper_limit)))

    for shift in range(angle_diff):
        for sign in [1, -1]:
            adjusted_angle = np.radians(base_angle + sign * shift)
            height, spacing = calculate_max_height(span, adjusted_angle, tol_round=tol_round)

            # If a suitable height is found
            if spacing:
                return height, spacing, base_angle + sign * shift
    return -1, -1, -1


def calculate_bridge(span: float, angle: int = 45, spacing: float = 0,
                     truss_mode: str = "pratt", lower_limit: int = 30,
                     upper_limit: int = 60, tol_round: Optional[Tuple[int, int]] = None) -> Tuple[float, float, float]:
    """
    Calculate the height of the bridge, spacing of columns, and diagonal length of rods.
    If spacing is not provided, the function will try to find a suitable height and
    spacing based on the angle: Priority: spacing -> angle

    Args:
        span (float): The total length of the bridge.
        angle (float): The angle of the diagonal rods in the bridge in 
                        degrees. Default is 45 degrees.

    Returns:
        (float, float, float): The height of the bridge, the distance between columns,
                               and the length of the diagonal elements (rods).
    """
    if not spacing:
        logging.info("Trying to find a suitable height and spacing based on the angle.")
        height, spacing, used_angle = try_angles(span, angle, lower_limit,
                                                 upper_limit, tol_round=tol_round)
        used_angle = round(used_angle, 2)

        if height == -1:
            raise RuntimeError("A suitable height for the bridge could not be found. Please adjust the span")

        if angle != used_angle:
            logging.warning("Adjusted angle to %.2f degrees to find a solution.", used_angle)
    else:
        height, spacing = calculate_max_height(span, np.radians(angle),
                                               spacing, tol_round=tol_round)
    # Divide spacing according to truss design
    space_divisor = 1 
    if truss_mode == "warren":
        space_divisor = 2 # Since nodes are in the middle of the beams
    diag = np.sqrt((spacing/space_divisor)**2 + height**2)
    return height, spacing, diag


def calculate_essential_elements(span: float, spacing: float, truss_mode: str ="pratt", 
                                 skip_rod: Optional[List] = None) -> Tuple[int, int, int, int, int, int]:
    """
    Calculate the number of columns, nodes, rods, beams and total elements.

    Args:
        span (float): The total length of the bridge.
        spacing (float): The distance between nodes (columns) in the bridge.
        truss_mode (str): The mode of the truss bridge (warren, pratt)
                        Defaults to pratt.

    Returns:
        (int, int, int, int, int): The number of columns, nodes, 
                                rods, beams, and total elements.
    """
    ratio = int(span // spacing)
    if skip_rod is None:
        skip_rod = []

    if truss_mode != "warren":
        n_columns = ratio - 1
        n_beams = int(ratio * 2 - 2)
        n_bot_beams = int(n_beams // 2 + 1)
        n_rods = n_beams
    else:
        n_columns = 0
        n_beams = int(ratio * 2 - 1)
        n_bot_beams = int(np.ceil(n_beams / 2))
        n_rods = n_beams + 1 

    n_rods = n_rods - len(skip_rod)
    n_nod_tot = n_beams + 2
    n_ele_tot = n_columns + n_rods + n_beams

    return n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams


def calculate_simple_elements(span: float, spacing: float, truss_mode: str, col_placements: Optional[List] = None,
                              skip_col: Optional[List] = None, beam_partition: int = 1) -> Tuple[int, int, int, int, int, int]:
    """
    Calculate the number of columns and beams for a simple bridge, along with their spacing.

    Args:
        span (float): The total length of the bridge.
        spacing (float): The distance between nodes (columns) in the bridge.
        truss_mode (str): The mode of the truss bridge (simple, simple_cant)
                        Defaults to pratt.

    Returns:
        (int, int, int, int, int): The number of columns, nodes, 
                                rods, beams, and total elements.
    """
    extra_beams = 0 # Extra beams for cantilevered bridges
    if col_placements:
        n_columns = len(col_placements)
    else:
        n_columns = int(span // spacing) + 1
    
    if skip_col:
        n_columns -= len(skip_col)

    if truss_mode != "simple":
        extra_beams = 1

    if beam_partition > 1:
        if col_placements:
            raise ValueError("Cannot partition beams with custom column placements.")
        if spacing // beam_partition < spacing / beam_partition:
            raise ValueError("Beam partition should be a divisor of the spacing.")
        n_beams = (n_columns - 1 + extra_beams) * beam_partition
    else:
        print("Defaulting partition to 1.")
        n_beams = n_columns - 1 + extra_beams
    
    n_nod_tot = n_beams + 2
    n_ele_tot = n_columns + n_beams

    return n_columns, n_nod_tot, 0, n_beams, n_ele_tot, 0


def calculate_element_node(span: float, spacing: float, height: float, n_dim: int,
                           n_par_nod: int, truss_mode: str = "pratt",
                           skip_rod: Optional[List] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate the nodal coordinates, nodal-param relation and 
    element-node relationships for a truss bridge.

    Args:
        span (float): The total length of the bridge.
        spacing (float): The distance between nodes (columns) in the bridge.
        height (float): The height of the bridge.
        diag (float): The length of the diagonal elements (rods) in the bridge.
        n_dim (int): The number of dimensions in the bridge 
                    (usually 2 for 2D bridges).
        n_par_nod (int): The number of parameters per node 
                        (usually 2 for 2D bridges: x and y coordinates).

    Returns:
        nodal_coord (numpy.ndarray): A 2D array where each row represents a 
                                    node and the columns are the x and y coordinates of the node.
        par (numpy.ndarray): A 2D array where each row represents a node and 
                            the columns are the parameters associated with the node.
        pel (numpy.ndarray): A 2D array where each row represents an element (beam, column, or rod) 
                            and the columns are the nodes associated with the element.
        ele_nod (numpy.ndarray): A 2D array where each row represents an element and 
                                the columns are the nodes associated with the element.
        n_par_tot (int): The total number of parameters in the bridge.
    """
    # Calculate the number of columns, rods, beams, total nodes and total parameters
    if skip_rod is None:
        skip_rod = []
        
    n_columns, n_nod_tot, n_rods, n_beams, n_ele_tot, n_bot_beams = calculate_essential_elements(span, spacing, truss_mode)
    n_par_tot = n_nod_tot * n_par_nod

    # Calculate the positions of the nodes
    nodal_coord = nodal_coords(n_nod_tot, n_dim, n_columns, spacing, height, n_bot_beams, truss_mode)

    # Calculate the nodal-param relation
    par = np.arange(1, n_nod_tot * n_par_nod + 1).reshape(n_nod_tot, n_par_nod)

    # Calculate the element-param relationships
    pel = pel_ele(par, n_columns, n_beams, n_rods, n_par_nod, n_nod_tot, 
                  n_ele_tot, n_bot_beams, truss_mode, skip_rod)

    # Calculate the element-node relationships
    ele_nod = fill_ele_nod(n_ele_tot, n_par_nod, pel, skip_rod)
    
    return nodal_coord, par, pel, ele_nod, n_par_tot


def calculate_element_properties(n_ele_tot: int, n_columns: int, n_beams: int, diag: float, spacing: float, 
                                 height: float, J: np.array, A: np.array, h: np.array, beta: np.array,
                                 ro: np.array, E: np.array, X: np.array, Y: np.array, ele_nod: List,
                                 shear_mod: int, width_properties: Dict, height_properties: Dict,
                                 unit_weight_properties: Dict, elastic_mod_properties: Dict,
                                 truss_mode: str, beam_partition: int = 1) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate the properties of the elements in the truss bridge.

    Returns:
        J, A, h, beta, ro, E: Numpy arrays of the moments of inertia, areas and heights, angles, 
        unit weights and elastic moduli of the elements. With J in m^4, A in m^2 and h in m, beta 
        in radians, ro in kN/m^3 and E in kN/m^2.
    """
    # Calculate the areas
    area_beam = width_properties['beam'] * height_properties['beam']
    area_column = width_properties['column'] * height_properties['column']
    area_rod = width_properties['rod'] * height_properties['rod']

    # Calculate the moments of inertia
    inertia_beam = width_properties['beam']**3 * height_properties['beam'] / 12
    inertia_column = width_properties['column']**3 * height_properties['column'] / 12
    inertia_rod = width_properties['rod']**3 * height_properties['rod'] / 12

    # Calculate the properties of the elements
    J[:n_beams] = inertia_beam
    A[:n_beams] = area_beam
    if "simple" in truss_mode:
        h[:n_beams] = spacing / beam_partition
    else:
        h[:n_beams] = spacing
    ro[:n_beams] = unit_weight_properties['beam']
    E[:n_beams] = elastic_mod_properties['beam']

    if n_beams != n_ele_tot:
        J[n_beams:n_beams + n_columns] = inertia_column
        A[n_beams:n_beams + n_columns] = area_column
        h[n_beams:n_beams + n_columns] = height
        ro[n_beams:n_beams + n_columns] = unit_weight_properties['column']
        E[n_beams:n_beams + n_columns] = elastic_mod_properties['column']

        J[n_beams + n_columns:] = inertia_rod
        A[n_beams + n_columns:] = area_rod
        h[n_beams + n_columns:] = diag
        ro[n_beams + n_columns:] = unit_weight_properties['rod']
        E[n_beams + n_columns:] = elastic_mod_properties['rod']
    
    # Debugging for any errors
    if np.any(J == 0) or np.any(A == 0) or np.any(h == 0) or np.any(ro == 0) or np.any(E == 0):
        raise ValueError("There are materials with no property")

    for i, _ in enumerate(beta):
        dx = X[ele_nod[i, 1]] - X[ele_nod[i, 0]]
        if abs(h[i]) < 1e-10:  # Avoid division by zero
            beta[i] = 0
        else:
            # Ensure value is within [-1, 1] for arccos
            cos_val = np.clip(dx / abs(h[i]), -1.0, 1.0)
            beta[i] = np.arccos(cos_val)

    G = np.full(len(E), shear_mod, dtype=np.float32)

    h = h.astype(np.float32)
    A = A.astype(np.float32)
    E = E.astype(np.float32)
    J = J.astype(np.float32)
    beta = beta.astype(np.float32)
    ro = ro.astype(np.float32)

    return J, A, h, beta, ro, E, G


def boundary_conditions(n_bot_beams: int, n_par_nod: int, n_nod_tot: int,
                        supports: Optional[List[str]] = None) -> np.ndarray:
    """
    Calculate the boundary conditions for the truss bridge.

    Returns:
        A numpy array where the ith element is 1 if the i-th parameter is a boundary condition and 0 otherwise.
        Defaults to pin and roller supports. Considers fixed supp if not a pin or roller.
    """
    if supports is None:
        supports = ["pin", "roller"]

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

    # Create the boundary array and set the boundary conditions
    temp = np.zeros(n_nod_tot * n_par_nod, dtype=np.int32)
    temp[:n_par_nod] = support_dof(supports[0], n_par_nod)
    temp[n_par_nod*n_bot_beams:n_par_nod*(n_bot_beams+1)] = support_dof(supports[1], n_par_nod)

    return temp


def truss_design(n_bot_beams: int, n_rods: int,
                 truss_mode: str ="pratt") -> np.ndarray:
    """
    Modify the number of rods to skip based on the design of the truss bridge.

    Returns:
        A numpy array of the rods to skip based on the truss design.
    """
    truss_mode = truss_mode.lower()
    def pratt_howe(n_bot_beams, n_rods, start=3, mid=0):
        left_side = np.arange(start, n_bot_beams, 2)
        right_side = np.arange(n_bot_beams+mid, n_rods, 2)
        return np.concatenate((left_side, right_side)).tolist()

    if truss_mode == "pratt":
        return pratt_howe(n_bot_beams-1, n_rods-1, 2)

    elif truss_mode == "howe":
        return pratt_howe(n_bot_beams-1, n_rods, 1, 1)

    else:
        return np.array([])


def col_pos(W: np.ndarray, n_par_nod: int, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the column positions for the truss bridge.

    Args:
        W: Boundary condition array (1 for column, 0 for no column)
        n_par_nod: Number of parameters per node
        X: X coordinates of nodes
        Y: Y coordinates of nodes (optional)

    Returns:
        A tuple of numpy arrays (X_col, Y_col) representing the x and y coordinates of the columns.
    """
    y_col = []
    x_col = []

    for i, pos in enumerate(X):
        if 1 in W[i*n_par_nod:(i+1)*n_par_nod]:  # Check if there is a column at this node
            x_col.append(pos)
            y_col.append(0)
            if Y is not None:
                y_col[-1] = Y[i]
    
    # Change to numpy arrays
    x_col = np.array(x_col)
    y_col = np.array(y_col)

    return x_col, y_col


################## Plotting Functions ##################
def plot_elements(ax, truss_mode, ele_nod, X, Y, h, beta):
    """
    Plot the structural elements (1D or 2D) on the given axes.

    Args:
        ax: Matplotlib axes object.
        truss_mode: String indicating the truss mode ("simple" for 1D, others for 2D).
        ele_nod: Numpy array of element-node connectivity.
        X: Numpy array of X coordinates of nodes.
        Y: Numpy array of Y coordinates of nodes.
        h: Numpy array of element lengths.
        beta: Numpy array of element angles in radians.
    """
    if "simple" in truss_mode:
        # Plot each element as a horizontal line
        for i in range(len(ele_nod)):
            x1 = X[ele_nod[i, 0]]  # Start node X
            length = h[i]
            x2 = x1 + length

            ax.plot([x1, x2], [0, 0], 'b')  # Plot element
            ax.text((x1 + x2) / 2, 0, str(i), verticalalignment='bottom')  # Label element

        ax.plot(X, np.zeros_like(X), 'ro')  # Plot nodes as red circles on the x-axis
        ax.set_title('1D Element Plot')
    else:
        # Plot each element based on angle beta
        for i in range(len(ele_nod)):
            x1, y1 = X[ele_nod[i, 0]], Y[ele_nod[i, 0]]  # Start node coordinates
            length = h[i]
            x2 = x1 + length * np.cos(beta[i])
            y2 = y1 + length * np.sin(beta[i])

            ax.plot([x1, x2], [y1, y2], 'b')  # Plot element
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(i))  # Label element

        ax.plot(X, Y, 'ro')  # Plot nodes
        ax.set_title('2D Element Plot')


def plot_supports(ax, X_col, Y_col, W, n_par_nod, X):
    """
    Plot supports (roller, pin, fixed) on the structure.

    Args:
        ax: Matplotlib axes object.
        X_col: Numpy array of X coordinates where supports are located.
        Y_col: Numpy array of Y coordinates where supports are located.
        W: Numpy array representing boundary conditions (1 for restrained, 0 for free).
        n_par_nod: Number of parameters per node.
        support_types: List of support types to cycle through.
    """
    for i, _ in enumerate(X_col):
        x = X_col[i]
        y = Y_col[i]
        
        # Find the node index
        idx = np.where(X == x)[0][0]
        support_type = sum(W[idx * n_par_nod:(idx + 1) * n_par_nod])

        if support_type == 1:
            # Larger circle for roller
            circle_radius = 0.03  # Increased size
            circle = plt.Circle((x, y - circle_radius), circle_radius, color='k', fill=True)
            ax.add_patch(circle)
        elif support_type == 2:
            # Larger triangle for pin
            triangle_base = 0.1
            triangle_height = 0.05

            tri_x = [x - triangle_base/2, x + triangle_base/2, x]
            tri_y = [y - triangle_height, y - triangle_height, y]
            ax.fill(tri_x, tri_y, 'k')
        elif support_type == 3:
            # Fixed support with lines
            line_height = 0.015
            line_width = 0.05

            ax.plot([x - line_width/2, x + line_width/2], 
                    [y - line_height, y - line_height], 'k', linewidth=2)
            for j in range(5):
                gap = line_width / 4
                ax.plot([x - line_width/2 + j * gap, x - line_width/2 + j * gap],
                        [y - line_height, y - line_height - 0.02], 'k')
