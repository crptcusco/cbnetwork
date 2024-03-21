import parsl
from parsl import python_app
from itertools import product
from concurrent.futures import as_completed

from classes.cbnetwork import CBN
from classes.pathcircletemplate import PathCircleTemplate

parsl.load()


def run_experiment():
    # Experiment parameters
    N_LOCAL_NETWORKS = 400
    N_VAR_NETWORK = 10
    N_OUTPUT_VARIABLES = 2
    N_INPUT_VARIABLES = 2
    V_TOPOLOGY = 4

    o_path_circle_template = PathCircleTemplate.generate_aleatory_template(
        n_var_network=N_VAR_NETWORK, n_input_variables=N_INPUT_VARIABLES)

    # Generate the CBN with o template
    o_cbn = o_path_circle_template.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                                              n_local_networks=N_LOCAL_NETWORKS)

    o_cbn.show_cbn()

    # Run the find_local_attractors_parsl function in parallel
    tasks = CBN.find_local_attractors_parsl(o_cbn.l_local_networks)

    # Monitorizar el progreso
    total_tasks = len(tasks)
    completed_tasks = 0

    print("Progreso:")
    for completed_task in as_completed(tasks):
        completed_tasks += 1
        print(f"Tareas completadas: {completed_tasks}/{total_tasks}")

    # Wait for all tasks to complete
    o_cbn.l_local_networks = [task.result() for task in tasks]

    # Show local attractors after all tasks have completed
    o_cbn.show_local_attractors()

    print("END OF EXPERIMENT")

# Run the experiment
run_experiment()
