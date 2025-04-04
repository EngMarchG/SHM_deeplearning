# General parameters
span = 1.4 # in m
angle = 0 # in degrees
n_dim = 2 # Number of dimensions
n_nod_ele = 2 # Number of nodes per element
n_par_nod = 3 # Number of parameters per node
n_par_ele = n_par_nod * n_nod_ele # Number of parameters per element

# Plot settings
n_discritizations = 10 # Number of points for plotting
n_plots = 4 # Number of plots for bridge"s truss

# For paper verification, steel beams are 4x4 cm and concrete beams are 12x27 in cm 
width_beam = 0.04
height_beam = 0.04
width_beam_conc = 0.12
height_beam_conc = 0.27

width_column = 0.4
height_column = 0.4

width_rod = 0.1
height_rod = 0.1

# Horizontal load on columns and angle in kN and degrees
po = 100
theta = 0

# Unit weight and elastic modulus in kN/m^3 and kN/m^2
unit_weight_steel = 78.5 # Steel 78.5 typically
elastic_mod = 183*10**6 # Steel 210 GPA typically
elastic_mod_rod = 210*10**6

# Compressive 51 Mpa and tensile 3.6 Mpa
# Unit weight and elastic modulus in kN/m^3 and kN/m^2
unit_weight_conc = 18.7 # Steel 78.5, concrete 1870
elastic_mod_conc = 24*10**6 # Steel 210, concrete 24
elastic_mod_rod = 21*10**7

# Shear modulus in kN/m^2
shear_mod = 8*10**6
k_shear = 0.9



################## HASHMAPS ##################

# put them all in a hashmap for easy access
width_properties = {}
height_properties = {}
unit_weight_properties = {}
elastic_mod_properties = {}

width_properties["beam"] = width_beam
width_properties["column"] = width_column
width_properties["rod"] = width_rod

height_properties["beam"] = height_beam
height_properties["column"] = height_column
height_properties["rod"] = height_rod

unit_weight_properties["beam"] = unit_weight_steel
unit_weight_properties["column"] = unit_weight_steel
unit_weight_properties["rod"] = unit_weight_steel

elastic_mod_properties["beam"] = elastic_mod
elastic_mod_properties["column"] = elastic_mod
elastic_mod_properties["rod"] = elastic_mod_rod

