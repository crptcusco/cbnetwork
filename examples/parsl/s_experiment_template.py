# local imports
from classes.cbnetwork import CBN
from classes.pathcircletemplate import PathCircleTemplate

# libraries imports
import parsl
import logging
from parsl.dataflow.dflow import logger

# Parsl Configurations
logger.disabled = True
# config error messages
parsl.set_stream_logger(level=logging.ERROR)
# load Parsl configuration
parsl.load()


def run_experiment():
    # Experiment parameters
    N_LOCAL_NETWORKS = 10
    N_VAR_NETWORK = 5
    N_OUTPUT_VARIABLES = 2
    N_INPUT_VARIABLES = 2
    V_TOPOLOGY = 4

    o_path_circle_template = PathCircleTemplate.generate_aleatory_template(
        n_var_network=N_VAR_NETWORK, n_input_variables=N_INPUT_VARIABLES)

    # Generate the CBN with o template
    o_cbn = o_path_circle_template.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                                              n_local_networks=N_LOCAL_NETWORKS)

    # Show the CBN Information
    o_cbn.show_description()

    # Find local attractors parallelized with Parsl
    tasks1 = CBN.find_local_attractors_parsl(o_cbn.l_local_networks)
    # Wait for all tasks to complete
    o_cbn.l_local_networks = [task.result() for task in tasks1]

    # Process
    for o_local_network in o_cbn.l_local_networks:
        o_cbn.process_kind_signal(o_local_network)

    # Show local attractors after all tasks have completed
    o_cbn.show_local_attractors()

    # Find attractor pairs parallelized with Parsl
    tasks2 = CBN.find_compatible_pairs_parsl(o_cbn)
    # Wait for all tasks to complete
    o_cbn.l_directed_edges = [task.result() for task in tasks2]
    o_cbn.show_attractor_pairs()

    # Find and show stable attractor fields
    print("normal function")
    # o_cbn.find_attractor_fields()
    print("parallel function")
    o_cbn.mount_stable_attractor_fields_parsl()
    o_cbn.show_stable_attractor_fields()

    print("END OF EXPERIMENT")


# Run the experiment
run_experiment()

logger.disabled = False
