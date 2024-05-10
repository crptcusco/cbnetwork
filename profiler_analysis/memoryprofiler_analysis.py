import pickle

# select CBN object to evaluate the performance
# path_cbn = '5_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/cbn_9_3.pkl'
# path_cbn = '../experiments/structural/5_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_5_3.pkl'
path_cbn = '../experiments/structural/5_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_10_3.pkl'

with open(path_cbn, 'rb') as file:
    o_cbn = pickle.load(file)

# Show the object
print(o_cbn)

# Show the CBN Information
o_cbn.show_description()

# Find Local attractors
o_cbn.find_local_attractors_heap()
o_cbn.show_local_attractors()

# Find attractor pairs
o_cbn.find_compatible_pairs()
o_cbn.show_attractor_pairs()

# Find stable attractor fields, Run the profiler on the function
o_cbn.find_stable_attractor_fields_profile()
o_cbn.show_stable_attractor_fields()
