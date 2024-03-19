import parsl
from parsl import python_app
from itertools import product
from concurrent.futures import as_completed

from classes.cbnetwork import CBN
from classes.pathcircletemplate import PathCircleTemplate

parsl.load()


@python_app
def find_local_attractors_task(o_local_network, l_local_scenes):
    from classes.localscene import LocalScene
    from classes.localnetwork import LocalNetwork

    print('=' * 80)
    print("FIND ATTRACTORS FOR NETWORK:", o_local_network.index)
    if l_local_scenes is None:
        o_local_scene = LocalScene(index=1)
        o_local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(o_local_network, scene=None)
        o_local_network.l_local_scenes.append(o_local_scene)
    else:
        v_cont_index = 1
        for scene in l_local_scenes:
            o_local_scene = LocalScene(v_cont_index, scene, o_local_network.l_var_exterm)
            s_scene = ''.join(scene)
            o_local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(o_local_network, s_scene)
            o_local_network.l_local_scenes.append(o_local_scene)
            v_cont_index = v_cont_index + 1
    return o_local_network


def find_local_attractors_parsl(local_networks):
    tasks = []
    for local_network in local_networks:
        l_local_scenes = None
        if len(local_network.l_var_exterm) != 0:
            l_local_scenes = list(product(list('01'), repeat=len(local_network.l_var_exterm)))
        tasks.append(find_local_attractors_task(local_network, l_local_scenes))
    return tasks


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
    tasks = find_local_attractors_parsl(o_cbn.l_local_networks)

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
