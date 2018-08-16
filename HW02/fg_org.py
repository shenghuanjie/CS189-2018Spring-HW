import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

#data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
#data_x = data['x']
#data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]


def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    pass


def main():
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD, len(LAMBDA)))
    Evalid = np.zeros((KD, len(LAMBDA)))
    for D in range(KD):
        print(D)
        for i in range(len(LAMBDA)):
            print(LAMBDA[i])
            #Etrain[D, i], Evalid[D, i] = fit(D + 1, LAMBDA[i])

    print('Average train error:', Etrain, sep='\n')
    print('Average valid error:', Evalid, sep='\n')

    # YOUR CODE to find best D and i


if __name__ == "__main__":
    main()
