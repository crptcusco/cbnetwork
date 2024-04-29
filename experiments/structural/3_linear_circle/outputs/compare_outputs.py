import pickle

# Abre el archivo pickle en modo lectura binaria ('rb')
with open('2_0_data_slow/exp5_aleatory_linear_circle_3_3_10/pkl_cbn/cbn_10_3.pkl', 'rb') as f:
    # Carga los datos del archivo pickle
    o_cbn = pickle.load(f)

o_cbn.show_description()
o_cbn.show_stable_attractor_fields()
