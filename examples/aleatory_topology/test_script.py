from classes.cbnetwork import CBN
from classes.localtemplates import AleatoryTemplate

# parameters
n_local_networks = 6
n_var_network = 5
n_output_variables = 2
n_input_variables = 2
n_clauses_function = 2
v_topology = 2
n_edges = 10


o_local_template = AleatoryTemplate.generate_aleatory_template(n_var_network=n_var_network,
                                                               n_input_variables=n_input_variables,
                                                               n_output_variables=n_output_variables,
                                                               v_topology=v_topology)

o_cbn = CBN.generate_aleatory_cbn_by_topology(n_local_networks=n_local_networks, n_var_network=n_var_network,
                                              v_topology=v_topology, n_output_variables=n_output_variables,
                                              n_input_variables=n_input_variables, local_template=o_local_template,
                                              n_edges=n_edges)

o_cbn.plot_topology()
# o_local_template.show()
# o_cbn.show_description()

o_cbn.find_local_attractors_sequential()
o_cbn.show_local_attractors()

o_cbn.find_compatible_pairs()
o_cbn.show_attractor_pairs()

o_cbn.mount_stable_attractor_fields()
o_cbn.show_attractors_fields()

