# Frame constants
N_columns = 2
N_floors = 5
N_nod_tot = (N_floors + 1) * N_columns
N_par_nod = 3
N_par_tot = N_nod_tot * N_par_nod
N_ele_tot = N_floors * (2 * N_columns - 1)
N_nod_ele = 2
N_par_ele = N_par_nod * N_nod_ele
N_tot_bound = 6
N_plots = 4

# Number of points for plotting
N_discritizations = 10

# Distance between nodes in meters
X_dist = 4
Y_dist = 3

# Columns 40x40 and beams 30x35 in cm
width_beam = 0.3
height_beam = 0.35
width_column = 0.4
height_column = 0.4

# Horizontal load on columns and angle in kN and degrees
po = 100
theta = 0

# Unit weight and elastic modulus in kN/m^3 and kN/m^2
unit_weight = 78.5
elastic_mod = 21*10**7