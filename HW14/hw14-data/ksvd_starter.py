import numpy as np
from starter import *
import matplotlib.pyplot as plt
##################################################################################
####################################k-svd#########################################
##################################################################################

'''
    Providing the entire code for students to try out.

    Disclaimer: The code has not been debugged carefully and is only provided as 
    a starter code to play around.

'''

def update_dictionary_single(Z,X,D,k):
    """
    Single update of the code book
    Inputs: 
    Z: N * K
    X: N * d
    D: K * d
    Outpusts:
    The kth row of D.
    The kth column of Z. 
    """
    
    E_k = X - Z.dot(D) + np.expand_dims(Z[:,k], 1).dot(np.expand_dims(D[k],0))
    if sum(Z[:,k] != 0)> 0:
        E_kR = E_k[Z[:,k] != 0] 
        u,s,vT = np.linalg.svd(E_kR)
        v = vT.T
        D[k] = v[:,0] 
        Z[Z[:,k]!=0,k] = s[0] * u[:,0] 
        
    return D[k],Z[:,k]
    
def update_dictionary(Z,X,D):
    """
    This function updates the dictionary.
    Output: D
    """
    for k in range(len(Z[0])):
        D_k, Z_k = update_dictionary_single(Z,X,D,k)
        D[k] = D_k; Z[:,k] = Z_k
        
    return D
    
def ksvd_single(X,K,s, tol = 1e-6):
    """
    This function runs a single ksvd. 
    """
    init_idx = np.random.choice(len(X), size=K, replace=False, p=None) 
    D = X[init_idx]
    Z = sparse_coding(D,X,s)
    error = np.linalg.norm(X - Z.dot(D), ord='fro')**2/len(X)
    diff = tol + 1
    while diff > tol:
        Z_new = sparse_coding(D,X,s) 
        if np.linalg.norm(X - Z_new.dot(D), ord='fro') < np.linalg.norm(X - Z.dot(D), ord='fro'):
            Z = Z_new 
        D = update_dictionary(Z,X,D)

        new_error = np.linalg.norm(X - Z.dot(D), ord='fro')**2/len(X)
        
        diff = error - new_error
        error = new_error
        
    return D, Z, error
        
def ksvd(X,K,s, tol = 1e-6):
    """
    ksvd with multiple initializations. 
    """
    outputs = []
    errors = []
    for i in range(10):
        D,Z,error = ksvd_single(X,K,s, tol = tol)
        outputs.append([D,Z])
        errors.append(error)
    idx = np.argmax(errors)
    D,Z = outputs[idx]
    return D,Z

def ksvd_tmp(X,D,K,s,tol):
    Z = sparse_coding(D,X,s)  
    error = np.linalg.norm(X - Z.dot(D), ord='fro')**2/len(X)
    return error

# from ksvd import ApproximateKSVD 

# def ksvd(X,K,s): 
#     aksvd = ApproximateKSVD(n_components=s)
#     D = aksvd.fit(X).components_
#     Z = aksvd.transform(X) 
#     return D, Z 

##################################################################################
####################################test algorithm################################ 
##################################################################################
N = 100
d = 10
K = 10 
s = 2
X = generate_test_data(N, s, K, d) 
D_recover,Z_recover = ksvd(X, K, s)
error = compute_error(X,D_recover,Z_recover)
print('The error on the testing data is {}'.format(error))
##################################################################################
####################################Sparsity experiments#########################
##################################################################################
def exp_sparsity(X, K, title = 'zero-noise (K<d)', 
                 name = 'zero_noise_K_less_d.png',
                 ss = np.arange(1,5,1).astype(int)):
     
    errors = np.zeros(len(ss))
    for i,s_p in enumerate(ss):
        D_hat,Z_hat = ksvd(X, K, s_p)
        errors[i] = compute_error(X,D_hat,Z_hat)  

    plt.plot(ss,errors)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title(title)
    plt.xlabel('sparsity constraint s')
    plt.ylabel('reconstruction error')
    plt.savefig(name)
    plt.close()
     


##################################################################################
####################################Zero-noise data (K<d)#########################
##################################################################################
print('zero noise data with K<d...')
np.random.seed(0)
N = 200; s = 3; K = 5; d = 20; c = 0 
X,D,Z = generate_toy_data(N, s, K, d, c)
exp_sparsity(X, K, title = 'zero-noise (K<d)', 
                 name = 'zero_noise_K_less_d.png',
                 ss = np.arange(1,6,1).astype(int)) 

##################################################################################
####################################Zero-noise data (K>d)#########################
##################################################################################
print('zero noise data with K>d...')

np.random.seed(0)
N = 200; s = 2; K = 20; d = 5; c = 0 
X,D,Z = generate_toy_data(N, s, K, d, c)
exp_sparsity(X, K, title = 'zero-noise (K>d)', 
                 name = 'zero_noise_K_greater_d.png',
                 ss = np.arange(1,6,1).astype(int)) 


##################################################################################
####################################noisy data (K>d)#########################
##################################################################################
print('noisy data with K<d...')

np.random.seed(0)
N = 200; s = 3; K = 20; d = 5; c = 0.0001 
X,D,Z = generate_toy_data(N, s, K, d, c)
exp_sparsity(X, K, title = 'noise data (K>d)', 
                 name = 'noise_K_greater_d.png',
                 ss = np.arange(1,6,1).astype(int)) 

##################################################################################
####################################Plot atoms of dictionary###################### 
##################################################################################
N = 200; s = 2; K = 200; d = 100; c = 0.001 
X,D,Z = generate_toy_data(N, s, K, d, c)
for i in range(3):
    plt.plot(range(d), D[i])
plt.savefig('dictionary.png')
plt.close()









