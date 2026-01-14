# Example comparing brute-force and SAT-based attractor finders
from cbnetwork.localnetwork import LocalNetwork
from cbnetwork.internalvariable import InternalVariable


def extract_attractor_signatures(local_network):
    """Return a list of scenes where each scene is a sorted set of attractor tuples."""
    sigs = []
    for scene in local_network.local_scenes:
        scene_sigs = []
        for attr in scene.l_attractors:
            # Normalize to ints to compare outputs from both finders
            states = tuple(
                tuple(int(v) for v in s.l_variable_values) for s in attr.l_states
            )
            # Canonicalize cycle by rotation so lexicographically smallest state comes first
            if len(states) > 1:
                rotations = [tuple(states[i:] + states[:i]) for i in range(len(states))]
                states = min(rotations)
            scene_sigs.append(states)
        # sort attractors within a scene to make comparison order-independent
        sigs.append(tuple(sorted(scene_sigs)))
    return tuple(sigs)


def compare_network(net_builder):
    # Build two fresh instances for each method
    net_a = net_builder()
    net_b = net_builder()

    # Ensure total variables bookkeeping required by SAT-based routines
    for net in (net_a, net_b):
        net.total_variables = net.internal_variables.copy()
        net.total_variables_count = len(net.total_variables)
        net.cnf_variables_map = {}

    LocalNetwork.find_local_attractors_brute_force(net_a)
    LocalNetwork.find_local_attractors(net_b)

    sig_a = extract_attractor_signatures(net_a)
    sig_b = extract_attractor_signatures(net_b)

    print("Brute-force signatures:", sig_a)
    print("SAT-based signatures:", sig_b)
    print("Equal:", sig_a == sig_b)
    return sig_a == sig_b


def build_cycle_network():
    # 2-variable network that produces a 4-cycle (see tests)
    net = LocalNetwork(index=1, internal_variables=[1, 2])
    # Represent functions in CNF-list form compatible with SAT formulation
    # var1 = not 2  -> CNF: [[-2]]
    # var2 = 1 (means var2_next = var1) -> CNF: [[1]]
    var1 = InternalVariable(index=1, cnf_function=[[-2]])
    var2 = InternalVariable(index=2, cnf_function=[[1]])
    net.descriptive_function_variables = [var1, var2]
    return net


def build_fixed_network():
    # 1-variable network with fixed point(s)
    net = LocalNetwork(index=2, internal_variables=[1])
    # var1_next = var1 (identity) -> CNF: [[1]]
    var1 = InternalVariable(index=1, cnf_function=[[1]])
    net.descriptive_function_variables = [var1]
    return net


if __name__ == "__main__":
    print("Comparing 2-variable cycle network:")
    ok1 = compare_network(build_cycle_network)

    print("\nComparing 1-variable fixed network:")
    ok2 = compare_network(build_fixed_network)

    if ok1 and ok2:
        print("All comparisons matched.")
    else:
        print("Discrepancy found. See outputs above.")
