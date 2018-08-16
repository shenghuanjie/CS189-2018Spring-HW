from common import *
import numpy as np


def bmatrix(a):
    """Returns a LaTeX bmatrix
    Retrieved from https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\[']
    rv += [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    rv += [r'\]\par']
    return '\n'.join(rv)


########################################################################
######### Part b ###################################
########################################################################

########################################################################
#########  Gradient Computing and MLE ###################################
########################################################################
def compute_gradient_of_likelihood(single_obj_loc,
                                   sensor_loc, single_distance):
    """
    Compute the gradient of the loglikelihood function for part a.

    Input:
    single_obj_loc: 1 * d numpy array.
    Location of the single object.

    sensor_loc: k * d numpy array.
    Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: d-dimensional numpy array.

    """
    grad = np.zeros_like(single_obj_loc)
    # Your code: implement the gradient of loglikelihood
    for ik in range(sensor_loc.shape[0]):
        grad = grad + (2 * (1 - single_distance[ik] / np.linalg.norm(single_obj_loc - sensor_loc[ik, :]))
                         * (single_obj_loc - sensor_loc[ik, :]))
    return grad


def find_mle_by_grad_descent_part_b(initial_obj_loc,
                                    sensor_loc, single_distance, lr=0.001, num_iters=10000):
    """
    Compute the gradient of the loglikelihood function for part a.

    Input:
    initial_obj_loc: 1 * d numpy array.
    Initialized Location of the single object.

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """
    obj_loc = initial_obj_loc
    # Your code: do gradient descent
    for i in range(num_iters):
        obj_loc = obj_loc - (lr * compute_gradient_of_likelihood(obj_loc, sensor_loc, single_distance))
    return obj_loc


if __name__ == "__main__":
    ########################################################################
    #########  MAIN ########################################################
    ########################################################################

    # Your code: set some appropriate learning rate here  0.01, 0.001, 0.0001
    all_step_sizes = [1.0, 0.01, 0.001, 0.0001]
    for _, lr in enumerate(all_step_sizes):
        print('\hfill \linebreak')
        print('lr = ' + str(lr) + ' \par')
        np.random.seed(0)
        sensor_loc = generate_sensors()
        obj_loc, distance = generate_data(sensor_loc)
        single_distance = distance[0]
        print('The real object location is \par')
        print(bmatrix(obj_loc))
        # Initialized as [0,0]
        initial_obj_loc = np.array([[0., 0.]])
        estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc,
                                                            sensor_loc, single_distance, lr=lr, num_iters=10000)
        print('The estimated object location with zero initialization is \par')
        print(bmatrix(estimated_obj_loc))

        # Random initialization.

        initial_obj_loc = np.random.randn(1, 2)
        estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc,
                                                            sensor_loc, single_distance, lr=lr, num_iters=10000)
        print('The estimated object location with random initialization is \par')
        print(bmatrix(estimated_obj_loc))

