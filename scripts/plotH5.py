import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rv_data(time, rv, uncertainties, title):
    plt.figure(figsize=(12, 6))
    plt.errorbar(time, rv, yerr=uncertainties, fmt='o', markersize=3, elinewidth=1, capsize=2, alpha=0.7, label='RV Data')
    plt.xlabel('Time (days)')
    plt.ylabel('Radial Velocity (m/s)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to the project root

    file_path = os.path.join(project_root, 'data', 'training_data_2.h5')

    print(f"Reading file: {file_path}")

    with h5py.File(file_path, 'r') as hdf:
        star_systems = list(hdf.keys())
        print(f"Number of star systems: {len(star_systems)}")

        while True:
            # Ask the user to select a star system
            print("\nAvailable options:")
            print("- Enter a number between 0 and", len(star_systems) - 1, "to select a star system")
            print("- Enter 'list' to see the first 10 star system names")
            print("- Enter 'exit' to quit")

            user_input = input("Enter your choice: ").strip().lower()

            if user_input == 'exit':
                break
            elif user_input == 'list':
                print("First 10 star systems:", star_systems[:10])
                continue

            try:
                if user_input.isdigit():
                    star_index = int(user_input)
                    if star_index < 0 or star_index >= len(star_systems):
                        print("Invalid index. Please try again.")
                        continue
                    star_name = star_systems[star_index]
                else:
                    if user_input not in star_systems:
                        print("Invalid star system name. Please try again.")
                        continue
                    star_name = user_input

                star_group = hdf[star_name]

                # Load the RV data and time directly from the HDF5 file
                rv = star_group['radial_velocity'][:]
                uncertainties = star_group['uncertainties_data'][:]
                time = star_group['time'][:]

                # Get star mass and planet count
                star_mass = star_group.attrs['star_mass']
                planet_count = star_group.attrs['planet_count']

                # Plot the data without filtering
                title = f"RV Data for {star_name}\nStar Mass: {star_mass:.2e} kg, Planets: {planet_count}"
                plot_rv_data(time, rv, uncertainties, title)

            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again.")

    print("Exiting the script.")

if __name__ == "__main__":
    main()
