import pickle

# Leer el objeto desde el archivo pickle
with open("o_cbn.pkl", "rb") as file:
    o_aleatory_cbn = pickle.load(file)

for o_directed_edge in o_aleatory_cbn.l_directed_edges:
    print(o_directed_edge.index_variable)


l_global_scenes = []

# for key, o_attractor_field in o_aleatory_cbn.d_attractor_fields.items():
#     print(key, " : ", o_attractor_field)
#     # search the o_atractor for every attractor field
#     for i_attractor in o_attractor_field:
#         print(i_attractor)
#         o_attractor = o_aleatory_cbn.get_local_attractor_by_index(i_attractor)
#         o_attractor.show()


