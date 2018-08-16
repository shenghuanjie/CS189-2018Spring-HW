import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline


def ground_truth(n, d, s):
    """
    Input: Two positive integers n, d. Requires n >= d >=s. If d<s, we let s = d
    Output: A tuple containing i) random matrix of dimension n X d with orthonormal columns. and
             ii) a d-dimensional, s-sparse wstar with (large) Gaussian entries
    """
    if d > n:
        print("Too many dimensions")
        return None

    if d < s:
        s = d
    A = np.random.randn(n, d)  # random Gaussian matrix
    U, S, V = np.linalg.svd(A, full_matrices=False)  # reduced SVD of Gaussian matrix
    wstar = np.zeros(d)
    wstar[:s] = 10 * np.random.randn(s)

    np.random.shuffle(wstar)
    return U, wstar


def get_obs(U, wstar):
    """
    Input: U is an n X d matrix and wstar is a d X 1 vector.
    Output: Returns the n-dimensional noisy observation y = U * wstar + z.
    """
    n, d = np.shape(U)
    z = np.random.randn(n)  # i.i.d. noise of variance 1
    y = np.dot(U, wstar) + z
    return y


def LS(U, y):
    """
    Input: U is an n X d matrix with orthonormal columns and y is an n X 1 vector.
    Output: The OLS estimate what_{LS}, a d X 1 vector.
    """
    wls = np.dot(U.T, y)  # pseudoinverse of orthonormal matrix is its transpose
    return wls


def thresh(U, y, lmbda):
    """
    Input: U is an n X d matrix and y is an n X 1 vector; lambda is a scalar threshold of the entries.
    Output: The estimate what_{T}(lambda), a d X 1 vector that is hard-thresholded (in absolute value) at level lambda.
            When code is unfilled, returns the all-zero d-vector.
    """
    n, d = np.shape(U)
    wls = LS(U, y)
    what = np.zeros(d)

    # print np.shape(wls)
    ##########
    # TODO: Fill in thresholding function; store result in what
    #####################
    # YOUR CODE HERE:
    what[abs(wls) >= lmbda] = wls[abs(wls) >= lmbda]
    ###############

    return what


def topk(U, y, s):
    """
    Input: U is an n X d matrix and y is an n X 1 vector; s is a positive integer.
    Output: The estimate what_{top}(s), a d X 1 vector that has at most s non-zero entries.
            When code is unfilled, returns the all-zero d-vector.
    """
    n, d = np.shape(U)
    what = np.zeros(d)
    wls = LS(U, y)

    ##########
    # TODO: Fill in thresholding function; store result in what
    #####################
    # YOUR CODE HERE: Remember the absolute value!
    sorted_indices = np.argsort(abs(wls))
    what[sorted_indices[-s:]] = wls[[sorted_indices[-s:]]]
    ###############
    return what


def error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=True, true_s=5):
    """
    Plots the prediction error 1/n || U(what - wstar)||^2 = 1/n || what - wstar ||^2 for the three estimators
    averaged over num_iter experiments.

    Input:
    Output: 4 arrays -- range of parameters, errors of LS, topk, and thresh estimator, respectively. If thresh and topk
            functions have not been implemented yet, then these errors are simply the norm of wstar.
    """
    wls_error = []
    wtopk_error = []
    wthresh_error = []

    if param == 'n':
        arg_range = np.arange(100, 2000, 50)
        lmbda = 2 * np.sqrt(np.log(d))
        for n in arg_range:
            U, wstar = ground_truth(n, d, s) if s_model else ground_truth(n, d, true_s)
            error_wls = 0
            error_wtopk = 0
            error_wthresh = 0
            for count in range(num_iters):
                y = get_obs(U, wstar)
                wls = LS(U, y)
                wtopk = topk(U, y, s)
                wthresh = thresh(U, y, lmbda)
                error_wls += np.linalg.norm(wstar - wls) ** 2
                error_wtopk += np.linalg.norm(wstar - wtopk) ** 2
                error_wthresh += np.linalg.norm(wstar - wthresh) ** 2
            wls_error.append(float(error_wls) / n / num_iters)
            wtopk_error.append(float(error_wtopk) / n / num_iters)
            wthresh_error.append(float(error_wthresh) / n / num_iters)

    elif param == 'd':
        arg_range = np.arange(10, 1000, 50)
        for d in arg_range:
            lmbda = 2 * np.sqrt(np.log(d))
            U, wstar = ground_truth(n, d, s) if s_model else ground_truth(n, d, true_s)
            error_wls = 0
            error_wtopk = 0
            error_wthresh = 0
            for count in range(num_iters):
                y = get_obs(U, wstar)
                wls = LS(U, y)
                wtopk = topk(U, y, s)
                wthresh = thresh(U, y, lmbda)
                error_wls += np.linalg.norm(wstar - wls) ** 2
                error_wtopk += np.linalg.norm(wstar - wtopk) ** 2
                error_wthresh += np.linalg.norm(wstar - wthresh) ** 2
            wls_error.append(float(error_wls) / n / num_iters)
            wtopk_error.append(float(error_wtopk) / n / num_iters)
            wthresh_error.append(float(error_wthresh) / n / num_iters)

    elif param == 's':
        arg_range = np.arange(5, 55, 5)
        lmbda = 2 * np.sqrt(np.log(d))
        for s in arg_range:
            U, wstar = ground_truth(n, d, s) if s_model else ground_truth(n, d, true_s)
            error_wls = 0
            error_wtopk = 0
            error_wthresh = 0
            for count in range(num_iters):
                y = get_obs(U, wstar)
                wls = LS(U, y)
                wtopk = topk(U, y, s)
                wthresh = thresh(U, y, lmbda)
                error_wls += np.linalg.norm(wstar - wls) ** 2
                error_wtopk += np.linalg.norm(wstar - wtopk) ** 2
                error_wthresh += np.linalg.norm(wstar - wthresh) ** 2
            wls_error.append(float(error_wls) / n / num_iters)
            wtopk_error.append(float(error_wtopk) / n / num_iters)
            wthresh_error.append(float(error_wthresh) / n / num_iters)

    return arg_range, wls_error, wtopk_error, wthresh_error


# nrange contains the range of n used, ls_error the corresponding errors for the OLS estimate
# nrange, ls_error, _, _ = error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=True, true_s=5)


########
# TODO: Your code here: call the helper function for d and s, and plot everything
########
#YOUR CODE HERE:
all_paras = ['n', 'd', 's']
for para in all_paras:
    arg_range, wls_error, _, _ = error_calc(num_iters=10, param=para, n=1000, d=100, s=5, s_model=True, true_s=5)
    plt.loglog(arg_range, wls_error, label='wls')
    plt.title('errors for the OLS estimate vs. ' + para)
    plt.xlabel(para)
    plt.ylabel('wls_error')
    plt.legend()
    plt.savefig('Figure_3a-' + para + '.png')
    plt.close()

# TODO: Part (b)
##############
#YOUR CODE HERE:
all_paras = ['n', 'd', 's']
for para in all_paras:
    arg_range, wls_error, wtopk_error, wthresh_error = error_calc(num_iters=10, param=para, n=1000, d=100, s=5, s_model=True, true_s=5)
    plt.loglog(arg_range, wls_error, label='wls')
    plt.loglog(arg_range, wtopk_error, label='wtopk')
    plt.loglog(arg_range, wthresh_error, label='wthresh')
    plt.title('errors vs. ' + para)
    plt.xlabel(para)
    plt.ylabel('error')
    plt.legend()
    plt.savefig('Figure_3b-' + para + '.png')
    plt.close()

# TODO: Part (c)
##############
#YOUR CODE HERE:
all_paras = ['n', 'd', 's']
for para in all_paras:
    arg_range, wls_error, wtopk_error, wthresh_error = error_calc(num_iters=10, param=para, n=1000, d=100, s=5, s_model=False, true_s=100)
    plt.loglog(arg_range, wls_error, label='wls')
    plt.loglog(arg_range, wtopk_error, label='wtopk')
    plt.loglog(arg_range, wthresh_error, label='wthresh')
    plt.title('errors vs. ' + para)
    plt.xlabel(para)
    plt.ylabel('error')
    plt.legend()
    plt.savefig('Figure_3c-' + para + '.png')
    plt.close()
