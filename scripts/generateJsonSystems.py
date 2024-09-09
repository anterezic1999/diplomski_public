import numpy as np
import json
import random
import os
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
import string

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory paths
synth_star_systems_directory = os.path.join(os.path.dirname(current_dir), 'val_star_systems')
os.makedirs(synth_star_systems_directory, exist_ok=True)

# Constants
mass_sun = 1.989e30
mass_earth = 5.972e24
mass_jupiter = 1.898e27

def custom_planet_count_distribution():
    counts = np.arange(2, 21)
    probabilities = np.exp(-0.2 * (counts - 6) ** 2)  # Gaussian-like distribution centered at 6
    probabilities /= probabilities.sum()  # Normalize to make it a probability distribution
    return np.random.choice(counts, p=probabilities)


# Bimodal distribution for planet masses
def bimodal_planet_mass_distribution(earthlike_ratio):
    choice = np.random.choice(['earthlike', 'massive'],
                              p=[earthlike_ratio, 1 - earthlike_ratio])  # Ratio Earth-like to massive
    if choice == 'earthlike':
        # Masses centered around Earth mass, with a log-normal distribution
        return np.random.lognormal(mean=np.log(1), sigma=0.85) * mass_earth
    else:
        # Masses centered around a fraction of Jupiter mass, with a log-normal distribution
        return np.random.lognormal(mean=np.log(0.6), sigma=0.85) * mass_jupiter


def generate_semi_major_axes(num_planets, min_sma, max_sma):
    mean = 1.614
    std = 0.5941
    def generate_spacing():
        spacing = norm.rvs(loc=mean, scale=std)
        while spacing < 1.1:
            spacing = norm.rvs(loc=mean, scale=std)
        return spacing

    # Generate semi-major axes
    smas = [min_sma]
    if num_planets > 0:
        for _ in range(num_planets):
            spacing = generate_spacing()
            next_sma = smas[-1] * spacing
            if min_sma < next_sma < max_sma:
                smas.append(next_sma)

        # Sort the SMAs in ascending order
        smas.sort()

        # Ensure exactly num_planets SMAs
        if len(smas) > num_planets:
            smas = smas[:num_planets]

    return smas

def generate_num_planets(mean, std, min):
    while True:
        # Generate a random number from a normal distribution
        num_planets = round(random.gauss(mean, std))

        # Retry if the number is less than min
        if num_planets > min:
            return num_planets

def generate_system_name():
    # Generate 5 random letters (upper and lowercase)
    letters = ''.join(random.choices(string.ascii_letters, k=5))

    # Generate 2 random numbers
    numbers = ''.join(random.choices(string.digits, k=2))

    # Combine all parts
    return f"{letters}-{numbers}"

# Function to generate a single star system
def generate_star_system():
    star_system = {}
    num_planets = generate_num_planets(7, 3.3, 1)

    sys_name = generate_system_name()

    star_mass_solar_masses = random.uniform(0.05, 2)
    star_mass = star_mass_solar_masses * mass_sun
    star_system['star_mass'] = star_mass
    star_system['planets'] = []

    modifier = random.uniform(0.01, 1.2)
    min_sma = star_mass_solar_masses * modifier  # In AU

    normalized_compactness_ratio = random.uniform(55, 85)
    compactness_ratio = normalized_compactness_ratio * star_mass_solar_masses
    max_sma = min_sma * compactness_ratio

    # Generate semi-major axes using the new function
    smas = generate_semi_major_axes(num_planets, min_sma, max_sma)

    # Random ratio of Earth-like to massive planets for this star system
    earthlike_ratio = np.random.uniform(0, 1)

    for i in range(len(smas)):
        if num_planets == 0:
            break
        planet = {}
        planet['name'] = f"{sys_name}{chr(ord('a') + i)}"

        # Mass of the planet using bimodal distribution with random ratio
        planet['mass'] = bimodal_planet_mass_distribution(earthlike_ratio)  # in kg

        # Semi-major axis 'a'
        planet['a'] = smas[i]

        # Orbital period 'P' using Kepler's third law: P^2 = a^3 / M_star
        planet['P'] = np.sqrt(planet['a'] ** 3 / (star_mass / mass_sun)) * 365.2422  # in days

        # Eccentricity 'e' using a Rayleigh distribution
        planet['e'] = min(np.random.rayleigh(0.1), 0.4)

        planet["phase_offset"] = np.random.uniform(0, 2 * np.pi)

        # Inclination 'i' mostly around 0 and almost never above 30 degrees
        inclination = np.random.exponential(5)
        while inclination > 30:
            inclination = np.random.exponential(5)
        planet['i'] = inclination

        # Argument of periapsis (angle for the eccentricity direction)
        planet['w'] = np.random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2π radians

        star_system['planets'].append(planet)

    return star_system
def plot_star_systems(star_systems):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        if i >= len(star_systems):
            ax.axis('off')
            continue

        star_system = star_systems[i]

        # Plot the star
        ax.scatter(0, 0, s=25, c='red', edgecolors='red', zorder=3)

        # Plot the planets
        for planet in star_system['planets']:
            a = planet['a']
            e = planet['e']
            i = np.radians(planet['i'])
            w = planet['w']
            phase_offset = planet['phase_offset']

            # Calculate the x and y coordinates of the planet's orbit
            theta = np.linspace(0, 2 * np.pi, 1000)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            x = r * (np.cos(w + theta) * np.cos(i) - np.sin(w + theta) * np.sin(i))
            y = r * (np.sin(w + theta) * np.cos(i) + np.cos(w + theta) * np.sin(i))

            # Plot the planet's orbit
            ax.plot(x, y, '-', color='lightgray', linewidth=0.5, zorder=1)

            # Plot the planet
            planet_pos = int(phase_offset / (2 * np.pi) * len(x))
            ax.scatter(x[planet_pos], y[planet_pos], s=20, c='blue', edgecolors='darkblue', zorder=2)

        # Set the axis limits
        max_a = max(planet['a'] for planet in star_system['planets'])
        ax.set_xlim(-max_a * 1.1, max_a * 1.1)
        ax.set_ylim(-max_a * 1.1, max_a * 1.1)

        # Set the axis scale to linear (not log)
        ax.set_xscale('linear')
        ax.set_yscale('linear')

        # Add axis labels
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')

        # Add a title for each subplot
        ax.set_title(f"")

        # Set aspect ratio to equal
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("planetary_systems.png", dpi=300)
    plt.show()

# Function to generate n star systems
def generate_n_star_systems(n):
    star_systems = []
    planet_count = {i: 0 for i in range(10000)}

    # Create a tqdm progress bar
    for i in tqdm(range(n), desc="Generating star systems", unit="system"):
        star_system = generate_star_system()
        star_systems.append(star_system)

        # Count the number of planets in this system
        num_planets = len(star_system['planets'])
        planet_count[num_planets] += 1

        # Save the star system to a JSON file
        file_path = os.path.join(synth_star_systems_directory, f'star_system_{i}.json')
        with open(file_path, 'w') as json_file:
            json.dump(star_system, json_file, indent=4)

    #plot_star_systems(star_systems)

    print(f"\nGenerated {n} star systems and saved them to {synth_star_systems_directory}")
    print("\nDistribution of planets across systems:")
    for num_planets, count in planet_count.items():
        if count > 0:
            print(f"Systems with {num_planets} planet{'s' if num_planets != 1 else ''}: {count}")


# Generate n star systems
n = 10000
generate_n_star_systems(n)
