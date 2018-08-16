""" Tools for calculating Gradient Descent for ||Ax-b||. """
import matplotlib.pyplot as plt
import numpy as np


def main():
    ################################################################################
    # TODO(student): Input Variables
    allAs = (np.array([[10, 0], [0, 1]]), np.array([[15, 8], [6, 5]]))
    #A = np.array([[10, 0], [0, 1]])  # do not change this until the last part
    b = np.array([4.5, 6])  # b in the equation ||Ax-b||
    total_step_count = 10000  # number of GD steps to take
    # step_size = lambda i: 1  # step size at iteration i
    step_names = ('step=1', 'step=(5/6)^i', 'step=1/(i+1)')
    save_names = ('fixed', 'adaptive', 'reciprocal')
    step_funcs = (lambda i: 1, lambda i: np.power(5/6, i), lambda i: 1/(i+1))  # step size at iteration i
    ################################################################################

    for iA, A in enumerate(allAs):
        print('A='+str(A))
        for i, (step_title, file_name, step_size) in enumerate(zip(step_names, save_names, step_funcs)):
            initial_position = np.array([0, 0])  # position at iteration 0
            # computes desired number of steps of gradient descent
            positions = compute_updates(A, b, initial_position, total_step_count, step_size, b)

            # print out the values of the x_i
            print(step_title)
            print('number of step: ' + str(positions.shape[0] - 1))
            print(positions)
            print('optimal point:' + str(np.dot(np.linalg.inv(A), b)))

            # plot the values of the x_i
            plt.figure()
            plt.title(step_title + 'A='+str(A))
            plt.scatter(positions[:, 0], positions[:, 1], c='blue')
            plt.scatter(np.dot(np.linalg.inv(A), b)[0],
                        np.dot(np.linalg.inv(A), b)[1], c='red')
            plt.plot()
            #plt.show()
            plt.savefig('Figure_2e_' + file_name + '-' + str(iA) + '.png')
            plt.close()


def compute_gradient(A, b, x):
    """Computes the gradient of ||Ax-b|| with respect to x."""
    return np.dot(A.T, (np.dot(A, x) - b)) / np.linalg.norm(np.dot(A, x) - b)


def compute_update(A, b, x, step_count, step_size):
    """Computes the new point after the update at x."""
    return x - step_size(step_count) * compute_gradient(A, b, x)


def compute_updates(A, b, p, total_step_count, step_size, optimal=[], error=0.01):
    """Computes several updates towards the minimum of ||Ax-b|| from p.

    Params:
        b: in the equation ||Ax-b||
        p: initialization point
        total_step_count: number of iterations to calculate
        step_size: function for determining the step size at step i
    """
    positions = [np.array(p)]
    for k in range(total_step_count):
        current_position = compute_update(A, b, positions[-1], k, step_size)
        positions.append(current_position)
        if (len(optimal) > 0) and (np.linalg.norm(current_position - optimal) <= error):
                break
    return np.array(positions)


#main()
#print(compute_gradient(np.array([[1, 0], [0, 1]]), np.array([4.5, 6]), np.array([4.8, 6.4])))

print(np.array([[1, 1, 2], [0, 1, 2]]).shape)