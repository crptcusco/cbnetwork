# external libraries
from itertools import product

# internal libraries
from classes.utils.customtext import CustomText


class GlobalNetwork:
    def __init__(self):
        self.l_variables = []
        self.d_functions = {}

    @staticmethod
    def generate_global_network(o_cbn):
        """
        Generate a global network ( Synchronous Boolean Network )
        :param o_cbn: Coupled Boolean Network Object
        :return: an object of Synchronous Boolean Network
        """
        l_variables = []
        for o_local_network in o_cbn.l_local_networks:
            for o_variable in o_local_network.l_variables:
                l_variables.append(o_variable)
        print(l_variables)

    def generate_global_scenes(self):
        CustomText.print_duplex_line()
        print("GENERATE GLOBAL SCENES")

        # get the index for every directed_edge
        l_global_signal_indexes = []
        for o_directed_edge in self.l_directed_edges:
            l_global_signal_indexes.append(o_directed_edge.index_variable)

        # generate the global scenes using all the combinations
        l_global_scenes_values = list(product(list('01'), repeat=len(self.l_directed_edges)))

        cont_index_scene = 1
        for global_scene_values in l_global_scenes_values:
            o_global_scene = GlobalScene(cont_index_scene, l_global_signal_indexes, global_scene_values)
            self.l_global_scenes.append(o_global_scene)
            cont_index_scene = cont_index_scene + 1

        CustomText.print_simple_line()
        print("Global Scenes generated")


class GlobalScene:
    def __init__(self, index, l_signal_indexes, l_values_signals):
        self.index = index
        self.l_signal_indexes = l_signal_indexes
        self.l_values_signals = l_values_signals

    def show(self):
        print("-------------------------------")
        print("Index Global Scene:", self.index)
        print("Indexes Directed Edges:", self.l_signal_indexes)
        print("Directed Edges Values:", self.l_values_signals)


class AttractorField:
    def __init__(self, index, l_attractor_indexes):
        self.index = index
        self.l_attractor_indexes = l_attractor_indexes
        # order the attractor indexes
        self.l_attractor_indexes.sort(key=lambda x: x.index)
        # calculate properties
        self.l_global_states = None

    def show(self):
        print("Attractor Field Index: ", self.index)
        print(self.l_attractor_indexes)

    @staticmethod
    def test_attractor_fields(o_cbn):
        b_flag = True
        for o_attractor_field in o_cbn.l_attractor_fields:
            if o_attractor_field.test_global_dynamic():
                print("Attractor Field", o_attractor_field.index, ": Passed")
            else:
                print("Attractor Field", o_attractor_field.index, ": Failed")
        return b_flag

    @staticmethod
    def generate_global_states(o_atractor_field, o_cbn):
        global_states = []
        for attractor_index in o_atractor_field.l_attractor_indexes:
            o_local_attractor = o_cbn.get_local_attractor_by_index(attractor_index)
            for o_state in o_local_attractor.states:
                global_states.append(o_state)

        o_atractor_field.l_global_states = global_states

    def test_global_dynamic(self):
        pass
        # for global_state in self.
        #     return True
        # return False

    def generate_global_dict_attractors(self):
        # For each local network we are gona generate a global index of attractor
        v_attractor_index = 1
        for o_local_network in self.l_local_networks:
            for o_local_scene in o_local_network.l_local_scenes:
                for o_attractor in o_local_network.l_attractors:
                    self.d_local_attractors[v_attractor_index] = (o_local_network.index,
                                                                  o_local_scene.index,
                                                                  o_attractor.index)

    def get_local_attractor_by_index(self, i_attractor):
        return self.d_local_attractors[i_attractor]
