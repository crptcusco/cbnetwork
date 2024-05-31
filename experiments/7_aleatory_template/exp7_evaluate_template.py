import pickle

from classes.cbnetwork import CBN
from classes.topologytemplate import TopologyTemplate

# Constants for the test
N_VAR_NETWORK = 5
N_LOCAL_NETWORKS = 6
V_TOPOLOGY = 6

# generate the aleatory local network template object
o_topology_template = TopologyTemplate.generate_aleatory_template(n_var_network=N_VAR_NETWORK, v_topology=6)
o_topology_template.show()

CBN.show_allowed_topologies()

# generate a linear CBN from the template
o_aleatory_cbn = o_topology_template.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                                                n_local_networks=N_LOCAL_NETWORKS)

o_aleatory_cbn.find_local_attractors_sequential()
o_aleatory_cbn.find_compatible_pairs()
o_aleatory_cbn.mount_stable_attractor_fields()

# generate a pickle object
# Open a file in binary write mode (wb)
pickle_path = "o_cbn.pkl"
with open(pickle_path, 'wb') as file:
    # Use pickle.dump to save the object to the file
    pickle.dump(o_aleatory_cbn, file)

# Close the file
file.close()
print("Pickle object saved in:", pickle_path)

o_aleatory_cbn.show_local_attractors()

print(o_aleatory_cbn.d_local_attractors)

# o_aleatory_cbn.show_attractor_pairs()
# o_aleatory_cbn.show_stable_attractor_fields()

