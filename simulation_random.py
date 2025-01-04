import numpy as np
import tensorflow as tf
import h5py
import os
import random
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, DirectivePattern
from sionna.channel.utils import cir_to_ofdm_channel
from filelock import FileLock

gpu_num = ""  # Use "" to use the CPU, use 0 to use the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

resolution = [480, 320]  # increase for higher quality of renderings


# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass


# Configure the notebook to use only a single GPU and allocate only as much memory as needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1)  # Set global random seed for reproducibility

gestures_dict = ['kucanie', 'lezenie_brzuch', 'stanie']

simulation_names = ['carla', 'claudia', 'eric', 'nathan']

dt = h5py.string_dtype(encoding='utf-8')

# Stwórz plik HDF5 lub otwórz, jeśli już istnieje
file_path = '/net/people/plgrid/plgkolaczb/Simulations/new_results_random.h5'
lock_path = file_path + '.lock'  # Tworzymy plik blokady
with FileLock(lock_path):
    with h5py.File(file_path, 'a') as f:
        if "combined_vectors" not in f:
            f.create_dataset("combined_vectors", shape=(0, 128), maxshape=(None, 128))
            print("Stworzono dataset 'combined_vectors' w głównym katalogu")
        else:
            print("Dataset 'combined_vectors' już istnieje")
        if "gesture_ids" not in f:
            f.create_dataset("gesture_ids", shape=(0,), maxshape=(None,))
            print("Stworzono dataset 'gesture_ids' w głównym katalogu")
        else:
            print("Dataset 'gesture_ids' już istnieje")

human_body = RadioMaterial("my_parameters_of_human_body",
                           relative_permittivity=52.0,
                           conductivity=1.73,
                           scattering_coefficient=0.9,
                           xpd_coefficient=0.02,
                           scattering_pattern=DirectivePattern(alpha_r=100))

for x1 in range(10000):
    i = random.randint(-350, 150)
    i = i / 100
    j = random.randint(-150, 250)
    j = j / 100
    edit_position = np.array([i, j, -30])
    r = random.randint(0, 7)
    edit_orientation = [np.pi * r / 4, 0, 0]
    gest = gestures_dict[random.randint(0, 2)]
    current_name = simulation_names[random.randint(0, 3)]
    if gest == 'kucanie':
        gesture_id = 2
    elif gest == 'lezenie_brzuch':
        gesture_id = 3
    elif gest == 'stanie':
        gesture_id = 6
    gesture_name = gest
    scene = load_scene(f'/net/people/plgrid/plgkolaczb/files/{current_name}/{gesture_name}.xml')
    actual_gesture = scene.get(name=f'{gesture_name}')

    # Center position
    position = actual_gesture.position
    centre_position = np.array([position[0], position[1], 0])
    actual_gesture.position = actual_gesture.position - centre_position

    # Create the antennas
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.7,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="VH")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=8,
                                 num_cols=2,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole",
                                 polarization="cross")
    tx_array = scene.tx_array
    rx_array = scene.rx_array

    # Set the frequency of the scene
    scene.frequency = 2.4e9
    scene.tx_array.frequency = 2.4e9
    scene.rx_array.frequency = 2.4e9

    f_c = scene.frequency

    # Create the transmitter and receiver using the arrays
    try:
        tx = Transmitter(name='Tx', position=[8.3, 5, 2.5], orientation=[np.pi / 2, 0, 0])
        rx = Receiver(name='Rx', position=[-8.3, -4.5, 5.0], orientation=[3 * np.pi / 2, 0, 0])
        scene.add(tx)
        scene.add(rx)
    except Exception as e:
        print(f"Error creating transmitter or receiver: {e}")
        raise

    scene.add(human_body)
    actual_gesture.radio_material = human_body
    actual_gesture.position = actual_gesture.position + edit_position
    actual_gesture.orientation = actual_gesture.orientation + edit_orientation

    paths = scene.compute_paths(scattering=True)
    a, tau = paths.cir()

    cfr = cir_to_ofdm_channel(scene.frequency, a, tau)

    amplitude = tf.abs(cfr)
    phase = tf.math.angle(cfr)

    reduced_amplitude = np.squeeze(amplitude.numpy()).reshape(1, -1)
    reduced_phase = np.squeeze(phase.numpy()).reshape(1, -1)

    amplitudes_array = np.array(reduced_amplitude)
    phases_array = np.array(reduced_phase)
    combined_vector = np.concatenate((amplitudes_array[-1], phases_array[-1]))

    with FileLock(lock_path):
        with h5py.File(file_path, 'a') as f:
            dset_combined_vector = f["combined_vectors"]
            dset_combined_vector.resize((dset_combined_vector.shape[0] + 1, 128))
            dset_combined_vector[-1:] = combined_vector
            dset_gesture_ids = f["gesture_ids"]
            dset_gesture_ids.resize((dset_gesture_ids.shape[0] + 1,))
            dset_gesture_ids[-1:] = gesture_id
