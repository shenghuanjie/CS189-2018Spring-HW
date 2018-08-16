import matplotlib.pyplot as plt
import numpy as np

from models import *
from starter import *


def main():
    np.random.seed(0)
    sensor_loc = generate_sensors()
    regular_loc, _ = generate_dataset(
        sensor_loc,
        num_sensors=sensor_loc.shape[0],
        spatial_dim=2,
        num_data=20,
        original_dist=True,
        noise=1)
    shifted_loc, _ = generate_dataset(
        sensor_loc,
        num_sensors=sensor_loc.shape[0],
        spatial_dim=2,
        num_data=20,
        original_dist=False,
        noise=1)

    plt.scatter(sensor_loc[:, 0], sensor_loc[:, 1], label="sensors")
    plt.scatter(regular_loc[:, 0], regular_loc[:, 1], label="regular points")
    plt.scatter(shifted_loc[:, 0], shifted_loc[:, 1], label="shifted points")
    plt.legend()
    plt.savefig("Figure_4a-dataset.png")
    # plt.show()


if __name__ == "__main__":
    main()
