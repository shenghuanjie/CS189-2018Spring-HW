import numpy as np
import matplotlib.pyplot as plt


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


# set a seed
np.random.seed(0)

X = np.array([[0, 0]])

while np.linalg.matrix_rank(X) != X.shape[1]:
    X = np.random.normal(scale = 20, size=(100,10))
print(np.linalg.matrix_rank(X)) # confirm that the matrix is full rank

# Theoretical optimal solution
w = np.random.normal(scale = 10, size = (10,1))
y = X.dot(w)

def sgd(X, y, w_actual, threshold, max_iterations, step_size, gd=False):
    if isinstance(step_size, float):
        step_size_func = lambda i: step_size
    else:
        step_size_func = step_size

    # run 10 gradient descent at the same time, for averaging purpose
    # w_guesses stands for the current iterates (for each run)
    n_runs = 10
    w_guesses = [np.zeros((X.shape[1], 1)) for _ in range(n_runs)]
    n = X.shape[0]
    error = []
    it = 0
    above_threshold = True
    previous_w = np.array(w_guesses)
    batch_size = 1  # as is pointed out on Piazza
    while it < max_iterations and above_threshold:
        it += 1
        curr_error = 0
        for j in range(len(w_guesses)):
            if gd:
                # Your code, implement the gradient for GD
                # sample_gradient = ?
                sample_gradient = 2 / X.shape[0] * (X.T.dot(X).dot(w_guesses[j]) - X.T.dot(y))
            else:
                # Your code, implement the gradient for SGD
                # sample_gradient =  ?
                sample_idxes = np.random.choice(X.shape[0], batch_size, replace=False)
                sampleX = X[sample_idxes, :]
                sampleY = y[sample_idxes, :]
                sample_gradient = 2 / batch_size * (sampleX.T.dot(sampleX).dot(w_guesses[j]) - sampleX.T.dot(sampleY))

            # Your code: implement the gradient update
            # learning rate at this step is given by step_size_func(it)
            # w_guesses[j] = ?
            w_guesses[j] -= step_size_func(it) * sample_gradient

            curr_error += np.linalg.norm(w_guesses[j]-w_actual)
        error.append(curr_error / n_runs)

        diff = np.array(previous_w) - np.array(w_guesses)
        diff = np.mean(np.linalg.norm(diff, axis=1))
        above_threshold = (diff > threshold)
        previous_w = np.array(w_guesses)
    return w_guesses, error


its = 5000
w_guesses, error = sgd(X, y, w, 1e-10, its, 0.0001)

plt.figure()
iterations = [i for i in range(len(error))]
#plt.semilogy(iterations, error, label = "Average error in w")
plt.semilogy(iterations, error, label = "Average error in w")
plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=True)
plt.title("Average Error vs Iterations for SGD with exact sol")
plt.legend()
#plt.show()
plt.savefig('Figure_2f-1.png')
plt.close()

print("Required iterations: " + str(len(error)) + '\par')
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses])
print("Final average error: ", bmatrix(average_error))

y2 = y + np.random.normal(scale=5, size = y.shape)
w=np.linalg.inv(X.T @ X) @ X.T @ y2


its = 5000
w_guesses2, error2 = sgd(X, y2, w, 1e-5, its, 0.0001)
w_guesses3, error3 = sgd(X, y2, w, 1e-5, its, 0.00001)
w_guesses4, error4 = sgd(X, y2, w, 1e-5, its, 0.000001)

w_guess_gd, error_gd = sgd(X, y2, w, 1e-5, its, 0.001, True)

plt.figure()
plt.semilogy([i for i in range(len(error2))], error2, label="SGD, lr = 0.0001")
plt.semilogy([i for i in range(len(error3))], error3, label="SGD, lr = 0.00001")
plt.semilogy([i for i in range(len(error4))], error4, label="SGD, lr = 0.000001")
plt.semilogy([i for i in range(len(error_gd))], error_gd, label="GD, lr = 0.001")
plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=True)
plt.title("Total Error vs Iterations for SGD without exact sol")
plt.legend()
#plt.show()
plt.savefig('Figure_2f-2.png')
plt.close()

print("Required iterations, lr = 0.0001: " + str(len(error2)) + '\par')
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses2])
print("Final average error: ", bmatrix(average_error))

print("Required iterations, lr = 0.00001: " + str(len(error3)) + '\par')
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses3])
print("Final average error: ", bmatrix(average_error))

print("Required iterations, lr = 0.000001: " + str(len(error4)) + '\par')
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses4])
print("Final average error: ", bmatrix(average_error))

print("Required iterations, GD: " + str(len(error_gd)) + '\par')
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guess_gd])
print("Final average error: ", bmatrix(average_error))

its = 5000
def step_size(step):
    if step < 500:
        return 1e-4
    if step < 1500:
        return 1e-5
    if step < 3000:
        return 3e-6
    return 1e-6

w_guesses_variable, error_variable = sgd(X, y2, w, 1e-10, its, step_size, False)

plt.figure()
plt.semilogy([i for i in range(len(error_variable))], error_variable, label="Average error, decreasing lr")
plt.semilogy([i for i in range(len(error2))], error2, label="Average error, lr = 0.0001")
plt.semilogy([i for i in range(len(error3))], error3, label="Average error, lr = 0.00001")
plt.semilogy([i for i in range(len(error4))], error4, label="Average error, lr = 0.000001")

plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=True)
plt.title("Error vs Iterations for SGD with no exact sol")
plt.legend()
#plt.show()
plt.savefig('Figure_2f-3.png')
plt.close()

print("Required iterations, variable lr: " + str(len(error_variable)) + '\par')
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses_variable])
print("Average error with decreasing lr:", bmatrix(average_error))
