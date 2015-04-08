import os, csv, glob, random
import numpy as np

def load_trajectory_by_path(path):
    positions = []
    if not os.path.isfile(path):
        raise ValueError('File does not exist')
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # Skip headers
        for row in reader:
            positions.append(map(float, row))
    positions = np.asarray(positions)

    return positions

def load_trajectory(driver, trajectory):
    path = 'data/drivers/%d/%d.csv' % (driver, trajectory)
    trajectory = load_trajectory_by_path(path)

    return trajectory

def load_random_trajectory():
    drivers = glob.glob('data/drivers/*')
    driver = random.choice(drivers)
    trajectories = glob.glob('%s/*.csv' % driver)
    path = random.choice(trajectories)
    print path
    trajectory = load_trajectory_by_path(path)

    return trajectory