import numpy as np
import tensorflow as tf
import h5py
import os
import random
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, DirectivePattern
from sionna.channel.utils import cir_to_ofdm_channel
from filelock import FileLock
from multiprocessing import Pool


def run_simulation(simulation_name, gesture, edit_positions, file_path, lock_path):
    gesture_id = gesture[0]
    gesture_name = gesture[1]

    # Load the scene
    scene = load_scene(f'/net/people/plgrid/plgkolaczb/files/{simulation_name}/{gesture_name}.xml')
    actual_gesture = scene.get(name=f'{gesture_name}')

    # Center position
    position = actual_gesture.position
    centre_position = np.array([position[0], position[1], 0])
    actual_gesture.position = actual_gesture.position - centre_position

    # Create antennas, set frequency, and add transmitter/receiver
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.7, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="VH")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=8, num_cols=2, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole",
                                 polarization="cross")
    tx_array = scene.tx_array
    rx_array = scene.rx_array

    # Set the frequency of the scene
    scene.frequency = 2.4e9
    scene.tx_array.frequency = 2.4e9
    scene.rx_array.frequency = 2.4e9

    # Create the transmitter and receiver using the arrays
    try:
        tx = Transmitter(name='Tx', position=[8.3, 5, 2.5], orientation=[np.pi / 2, 0, 0])
        rx = Receiver(name='Rx', position=[-8.3, -4.5, 5.0], orientation=[3 * np.pi / 2, 0, 0])
        scene.add(tx)
        scene.add(rx)
    except Exception as e:
        print(f"Error creating transmitter or receiver: {e}")
        raise

    human_body = RadioMaterial("my_parameters_of_human_body",
                               relative_permittivity=52.0,
                               conductivity=1.73,
                               scattering_coefficient=0.9,
                               xpd_coefficient=0.02,
                               scattering_pattern=DirectivePattern(alpha_r=100))
    scene.add(human_body)
    actual_gesture.radio_material = human_body

    for edit_position in edit_positions:
        if round(edit_position[0] * 100) % 15 == 0 and round(edit_position[1] * 100) == 248:
            with FileLock(f'/net/people/plgrid/plgkolaczb/Simulations/raport.txt.lock'):
                with open(f"/net/people/plgrid/plgkolaczb/Simulations/raport.txt", "a") as file:
                    file.write(
                        f"Proces {os.getpid()}, czyli dla: {simulation_name}_{gesture_name} wszedł do obliczeń {edit_position[0]} \n")
        # Adjust actual_gesture properties
        edit_orientation = [np.pi * (random.randint(0, 3)) / 2, 0, 0]
        actual_gesture.position = actual_gesture.position + edit_position
        actual_gesture.orientation = actual_gesture.orientation + edit_orientation

        # Compute paths and process results
        paths = scene.compute_paths(scattering=True)
        a, tau = paths.cir()
        cfr = cir_to_ofdm_channel(scene.frequency, a, tau)

        # Process amplitudes and phases
        amplitude = tf.abs(cfr)
        phase = tf.math.angle(cfr)

        reduced_amplitude = np.squeeze(amplitude.numpy()).reshape(1, -1)
        reduced_phase = np.squeeze(phase.numpy()).reshape(1, -1)

        combined_vector = np.concatenate((reduced_amplitude[-1], reduced_phase[-1]))

        # Save results to HDF5
        with FileLock(lock_path):
            with h5py.File(file_path, 'a') as f:
                dset_combined_vector = f["combined_vectors"]
                dset_combined_vector.resize((dset_combined_vector.shape[0] + 1, 128))
                dset_combined_vector[-1:] = combined_vector
                dset_gesture_ids = f["gesture_ids"]
                dset_gesture_ids.resize((dset_gesture_ids.shape[0] + 1,))
                dset_gesture_ids[-1:] = gesture_id


def main():
    # Set up parameters
    edit_positions = [np.array([i / 100, j / 100, -30]) for i in range(-349, 150, 4) for j in range(248, -150, -4)]
    # edit_orientations = [[np.pi * r / 2, 0, 0] for r in range(2)]

    simulation_names = ['carla', 'claudia', 'eric', 'nathan']
    gestures_dict = {(2, 'kucanie'), (3, 'lezenie_brzuch'), (6, 'stanie')}

    # Prepare arguments for each simulation
    tasks = []
    for simulation_name in simulation_names:
        for gesture in gestures_dict:
            file_path = f'/net/people/plgrid/plgkolaczb/Simulations/new_results_{simulation_name}_{gesture[1]}.h5'
            lock_path = file_path + '.lock'
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
            tasks.append((simulation_name, gesture, edit_positions, file_path, lock_path))

    # Use multiprocessing to run simulations
    with Pool(processes=12) as pool:
        pool.starmap(run_simulation, tasks)


if __name__ == "__main__":
    main()
