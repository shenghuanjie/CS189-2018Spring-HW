from common import *
from part_b_starter import compute_gradient_of_likelihood
from part_b_starter import find_mle_by_grad_descent_part_b
from part_c_starter import log_likelihood


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
#########  Gradient Computing and MLE ##################################
########################################################################
def compute_grad_likelihood(sensor_loc, obj_loc, distance):
    """
    Compute the gradient of the loglikelihood function for part f.

    Input:
    sensor_loc: k * d numpy array.
    Location of sensors.

    obj_loc: n * d numpy array.
    Location of the objects.

    distance: n * k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: k * d numpy array.
    """
    grad = np.zeros(sensor_loc.shape)
    # Your code: finish the grad loglike
    for ik in range(sensor_loc.shape[0]):
        for iN in range(obj_loc.shape[0]):
            grad[ik, :] += ((1 - distance[iN, ik] / np.linalg.norm(sensor_loc[ik, :] - obj_loc[iN, :]))
                            * (sensor_loc[ik, :] - obj_loc[iN, :]))
    grad *= 2
    return grad


def find_mle_by_grad_descent(initial_sensor_loc,
                             obj_loc, distance, lr=0.001, num_iters=1000):
    """
    Compute the gradient of the loglikelihood function for part f.

    Input:
    initial_sensor_loc: k * d numpy array.
    Initialized Location of the sensors.

    obj_loc: n * d numpy array. Location of the n objects.

    distance: n * k dimensional numpy array.
    Observed distance of the n object.

    Output:
    sensor_loc: k * d numpy array. The mle for the location of the object.

    """
    sensor_loc = initial_sensor_loc
    if lr == 0.05:
        print('distance.shape' + str(distance.shape))
        print('sensor_loc.shape'+str(sensor_loc.shape))
        print('obj_loc.shape' + str(obj_loc.shape))

    # Your code: finish the gradient descent
    for i in range(num_iters):
        # print(sensor_loc.shape)
        # print(compute_grad_likelihood(obj_loc, sensor_loc, distance).shape)
        sensor_loc -= (lr * compute_grad_likelihood(sensor_loc, obj_loc, distance))
    return sensor_loc


########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################

np.random.seed(0)
sensor_loc = generate_sensors()
obj_loc, distance = generate_data(sensor_loc, n=100)
print('The real sensor locations are')
print(bmatrix(sensor_loc))
# Initialized as zeros.
initial_sensor_loc = np.zeros((7, 2))  # np.random.randn(7,2)
estimated_sensor_loc = find_mle_by_grad_descent(initial_sensor_loc,
                                                obj_loc, distance, lr=0.001, num_iters=1000)
print('The predicted sensor locations are')
print(bmatrix(estimated_sensor_loc))


########################################################################
#########  Estimate distance given estimated sensor locations. ######### 
########################################################################

def compute_distance_with_sensor_and_obj_loc(sensor_loc, obj_loc):
    """
    estimate distance given estimated sensor locations.

    Input:
    sensor_loc: k * d numpy array.
    Location of the sensors.

    obj_loc: n * d numpy array. Location of the n objects.

    Output:
    distance: n * k dimensional numpy array.
    """
    estimated_distance = scipy.spatial.distance.cdist(obj_loc,
                                                      sensor_loc,
                                                      metric='euclidean')
    return estimated_distance


########################################################################
#########  MAIN  #######################################################
########################################################################    
np.random.seed(100)
########################################################################
#########  Case 1. #####################################################
########################################################################

mse = 0
Nrun = 100
Nrand = 10
for i in range(Nrun):
    obj_loc, distance = generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k=7, d=2)
    l = float('inf')
    # Your code: compute the mse for this case
    temp_mse = float('inf')
    for iRand in range(Nrand):
        initial_obj_loc = np.random.randn(1, 2)
        estimated_obj_loc = find_mle_by_grad_descent(initial_obj_loc,
                                                    estimated_sensor_loc,
                                                     distance.T, lr=0.001, num_iters=10000)
        if temp_mse > np.sum((estimated_obj_loc - obj_loc) ** 2):
            temp_mse = np.sum((estimated_obj_loc - obj_loc) ** 2)
    mse += temp_mse

print('The MSE for Case 1 is {}'.format(mse))

########################################################################
#########  Case 2. #####################################################
########################################################################
mse = 0
for i in range(Nrun):
    obj_loc, distance = generate_data(sensor_loc, k=7, d=2, n=1, original_dist=False)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k=7, d=2)
    l = float('-inf')
    # Your code: compute the mse for this case
    temp_mse = float('inf')
    for iRand in range(Nrand):
        initial_obj_loc = np.random.randn(1, 2)
        estimated_obj_loc = find_mle_by_grad_descent(initial_obj_loc, estimated_sensor_loc, distance.T, lr=0.001, num_iters=10000)
        if temp_mse > np.sum((estimated_obj_loc - obj_loc) ** 2):
            temp_mse = np.sum((estimated_obj_loc - obj_loc) ** 2)
    mse += temp_mse

print('The MSE for Case 2 is {}'.format(mse))

########################################################################
#########  Case 3. #####################################################
########################################################################
mse = 0

for i in range(Nrun):
    obj_loc, distance = generate_data(sensor_loc, k=7, d=2, n=1, original_dist=False)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k=7, d=2)
    l = float('-inf')
    # Your code: compute the mse for this case
    initial_obj_loc = np.array([[300.0, 300.0]],dtype=float)
    estimated_obj_loc = find_mle_by_grad_descent(initial_obj_loc, estimated_sensor_loc, distance.T, lr=0.001, num_iters=10000)
    mse += np.sum((estimated_obj_loc - obj_loc) ** 2)

print('The MSE for Case 2 (if we knew mu is [300,300]) is {}'.format(mse))
