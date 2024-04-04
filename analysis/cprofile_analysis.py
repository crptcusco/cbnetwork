# local imports
from classes.pathcircletemplate import PathCircleTemplate

# libraries imports
import cProfile
import pstats
import pickle

# Experiment parameters
N_LOCAL_NETWORKS = 10
N_VAR_NETWORK = 5
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2
V_TOPOLOGY = 4

# # Generate the template for the experiment
# o_path_circle_template = PathCircleTemplate.generate_aleatory_template(
#     n_var_network=N_VAR_NETWORK, n_input_variables=N_INPUT_VARIABLES)
#
# # Generate the CBN with o template
# o_cbn = o_path_circle_template.generate_cbn_from_template(v_topology=V_TOPOLOGY,
#                                                           n_local_networks=N_LOCAL_NETWORKS)


# path_cbn = '3_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/cbn_9_3.pkl'
# path_cbn = '../experiments/structural/3_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_5_3.pkl'
path_cbn = '../experiments/structural/3_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_10_3.pkl'

with open(path_cbn, 'rb') as file:
    o_cbn = pickle.load(file)

# Show the object
print(o_cbn)

# Show the CBN Information
o_cbn.show_description()

# Find Local attractors
o_cbn.find_local_attractors_optimized()
o_cbn.show_local_attractors()

# Find attractor pairs
o_cbn.find_compatible_pairs()
o_cbn.show_attractor_pairs()

# Find stable attractor fields
# o_cbn.find_stable_attractor_fields()
# Run the profiler on the function
cProfile.run('o_cbn.find_stable_attractor_fields()',
             'cprofile_stats')
o_cbn.show_stable_attractor_fields()

# Load the profile statistics
stats = pstats.Stats('cprofile_stats')
# Print the statistics
stats.print_stats()
print("Fin")


print("END OF EXPERIMENT")

