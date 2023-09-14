import heapq


class Node:
    def __init__(self, index, weight):
        self.index = index
        self.weight = weight

    def __lt__(self, other):
        # Define the comparison to sort nodes in the heap based on the weight
        return self.weight < other.weight


class CustomHeap:
    def __init__(self):
        self.heap = []

    def add_node(self, node):
        heapq.heappush(self.heap, node)

    def remove_node(self):
        if self.heap:
            return heapq.heappop(self.heap)
        else:
            return None

    def get_size(self):
        return len(self.heap)

    # @staticmethod
    # def calculate_weight(l_signals_status):
    #     # l_signals_status = [0, 0, 1, 2]
    #     res = sum(l_signals_status)
    #     return res

    # # Types Coupling Signals
    # kind_coupling_signals = {
    #     1: "not compute",
    #     2: "restricted",
    #     3: "stable",
    #     4: "not stable"
    # }

    # # How to compute the weight
    # dict_weight = {
    #     1: "stable",
    #     2: "not compute"
    # }

    # # Evaluate the signals that don't have input coupling signals
    # l_local_network_without_signals = []
    # for o_local_network in self.l_local_networks:
    #     if not o_local_network.l_input_signals:
    #         l_local_network_without_signals.append(o_local_network.index)
    # print(l_local_network_without_signals)

    # print(heap)
