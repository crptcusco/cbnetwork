import pickle

# Leer el objeto desde el archivo pickle
with open("o_cbn.pkl", "rb") as file:
    o_aleatory_cbn = pickle.load(file)

# o_aleatory_cbn.count_fields_by_global_scenes()
# generate the l_global_scenes
for o_directed_edge in o_aleatory_cbn.l_directed_edges:
    for key, value in o_directed_edge.d_out_value_to_attractor.items():
        print(key, "->", value)

for key, o_attractor_field in o_aleatory_cbn.d_attractor_fields.items():
    print(key, " : ", o_attractor_field)

print(o_aleatory_cbn.d_local_attractors)
for key, o_attractor in o_aleatory_cbn.d_local_attractors.items():
    print(key, "->", o_attractor)

# self.show_global_scenes()
l_global_scenes = []
for key, o_attractor_field in o_aleatory_cbn.d_attractor_fields.items():
    print("Attractor field:", key)
    l_scenes = []
    for i_attractor in o_attractor_field:
        l_scenes.append(o_aleatory_cbn.get_local_attractor_by_index(i_attractor).local_scene)
    print(l_scenes)
    l_global_scenes.append(l_scenes)
    l_global_scenes = list(set(l_global_scenes))
print(len(l_global_scenes))

