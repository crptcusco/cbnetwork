import pickle

# path_cbn = '3_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/cbn_9_3.pkl'
path_cbn = '3_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/cbn_9_4.pkl'

with open(path_cbn, 'rb') as file:
    o_cbn = pickle.load(file)

# Show the object
print(o_cbn)
o_cbn.show_cbn()

# show pairs
# o_cbn.show_attractor_pairs()

# # find attractor fields
# o_cbn.find_attractor_fields()
# # show the attractors
# o_cbn.show_attractors_fields()
