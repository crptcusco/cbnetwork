import cProfile
import pickle
import pstats

from classes.cbnetwork import CBN


# select the most cost cbn
# path_cbn = '3_linear_circle/outputs/exp5_aleatory_linear_circle_8_8_10/cbn_9_3.pkl'
path_cbn = 'exp5_aleatory_linear_circle_7_7_10/pkl_cbn/cbn_8_3.pkl'

with open(path_cbn, 'rb') as file:
    o_cbn = pickle.load(file)

# Show the object
o_cbn.show_description()

# # Perfilar la función
cProfile.run('CBN.find_local_attractors_parsl(o_cbn)', filename='stats.txt')

# Cargar el archivo de estadísticas generado por cProfile
stats = pstats.Stats('stats.txt')

# Ordenar las estadísticas por tiempo total
stats.sort_stats('tottime')

# Imprimir las 10 funciones principales por tiempo total
stats.print_stats(10)
