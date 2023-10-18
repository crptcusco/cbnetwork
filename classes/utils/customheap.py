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

    def get_indexes(self):
        indexes = []
        for node in self.heap:
            indexes.append(node.i_local_net)
        return indexes

    def update_node(self, index, new_weight):
        # Find the node with the specified index
        for i, node in enumerate(self.heap):
            if node.i_local_net == index:
                # Update the weight of the node
                node.weight = new_weight

                # Reorganize the heap to maintain the heap property
                heapq.heapify(self.heap)
