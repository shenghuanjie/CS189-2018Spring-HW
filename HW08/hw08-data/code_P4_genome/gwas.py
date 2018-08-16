import numpy as np
import scipy as sp
import scipy.stats as st
import scipy.interpolate
import scipy.linalg as la
import pylab as pl
import math
from scipy.stats import multivariate_normal
from sklearn import preprocessing
import statsmodels.api as sm
import pickle


import scipy as SP
import scipy.optimize as opt


def minimize1D(f, evalgrid = None, nGrid=10, minval=0.0, maxval = 0.99999, verbose=False, brent=True,check_boundaries = True, resultgrid=None, return_grid=False):
    '''
    minimize a function f(x) in the grid between minval and maxval.
    The function will be evaluated on a grid and then all triplets,
    where the inner value is smaller than the two outer values are optimized by
    Brent's algorithm.
    --------------------------------------------------------------------------
    Input:
    f(x)    : callable target function
    evalgrid: 1-D array prespecified grid of x-values
    nGrid   : number of x-grid points to evaluate f(x)
    minval  : minimum x-value for optimization of f(x)
    maxval  : maximum x-value for optimization of f(x)
    brent   : boolean indicator whether to do Brent search or not.
              (default: True)
    --------------------------------------------------------------------------
    Output list:
    [xopt, f(xopt)]
    xopt    : x-value at the optimum
    f(xopt) : function value at the optimum
    --------------------------------------------------------------------------
    '''
    #evaluate the target function on a grid:
    if verbose: print("evaluating target function on a grid")
    if evalgrid is not None and brent:# if brent we need to sort the input values
        i_sort = evalgrid.argsort()
        evalgrid = evalgrid[i_sort]
    if resultgrid is None:
        [evalgrid,resultgrid] = evalgrid1D(f, evalgrid = evalgrid, nGrid=nGrid, minval=minval, maxval = maxval  )
    
    i_currentmin=resultgrid.argmin()
    minglobal = (evalgrid[i_currentmin],resultgrid[i_currentmin])
    if brent:#do Brent search in addition to rest? 
        if check_boundaries:
            if verbose: print("checking grid point boundaries to see if further search is required")
            if resultgrid[0]<resultgrid[1]:#if the outer boundary point is a local optimum expand search bounded between the grid points
                if verbose: print("resultgrid[0]<resultgrid[1]--> outer boundary point is a local optimum expand search bounded between the grid points")
                minlocal = opt.fminbound(f,evalgrid[0],evalgrid[1],full_output=True)
                if minlocal[1]<minglobal[1]:
                    if verbose: print("found a new minimum during grid search")
                    minglobal=minlocal[0:2]
            if resultgrid[-1]<resultgrid[-2]:#if the outer boundary point is a local optimum expand search bounded between the grid points
                if verbose: print("resultgrid[-1]<resultgrid[-2]-->outer boundary point is a local optimum expand search bounded between the grid points")
                minlocal = opt.fminbound(f,evalgrid[-2],evalgrid[-1],full_output=True)
                if minlocal[1]<minglobal[1]:
                    if verbose: print("found a new minimum during grid search")
                    minglobal=minlocal[0:2]
        if verbose: print("exploring triplets with brent search")
        onebrent=False
        for i in range(resultgrid.shape[0]-2):#if any triplet is found, where the inner point is a local optimum expand search
            if (resultgrid[i+1]<resultgrid[i+2]) and (resultgrid[i+1]<resultgrid[i]):
                onebrent=True
                if verbose: print("found triplet to explore")
                minlocal = opt.brent(f,brack = (evalgrid[i],evalgrid[i+1],evalgrid[i+2]),full_output=True)
                if minlocal[1]<minglobal[1]:
                    minglobal=minlocal[0:2]
                    if verbose: print("found new minimum from brent search")
    if return_grid:
        return (minglobal[0], minglobal[1], evalgrid, resultgrid)
    else:
        return minglobal


def evalgrid1D(f, evalgrid = None, nGrid=10, minval=0.0, maxval = 0.99999, dimF=0):
    '''
    evaluate a function f(x) on all values of a grid.
    --------------------------------------------------------------------------
    Input:
    f(x)    : callable target function
    evalgrid: 1-D array prespecified grid of x-values
    nGrid   : number of x-grid points to evaluate f(x)
    minval  : minimum x-value for optimization of f(x)
    maxval  : maximum x-value for optimization of f(x)
    --------------------------------------------------------------------------
    Output:
    evalgrid    : x-values
    resultgrid  : f(x)-values
    --------------------------------------------------------------------------
    '''
    if evalgrid is None:
        step = (maxval-minval)/(nGrid)
        evalgrid = SP.arange(minval,maxval+step,step)
    if dimF:
        resultgrid = SP.ones((evalgrid.shape[0],dimF))*9999999999999.0
    else:
        resultgrid = SP.ones(evalgrid.shape[0])*9999999999999.0
    for i in range(evalgrid.shape[0]):        
        fevalgrid = f(evalgrid[i])
        assert SP.isreal(fevalgrid).all(),"function returned imaginary value"
        resultgrid[i] = fevalgrid
    return (evalgrid,resultgrid)


# **********************************************************************

#took out:
# lower = -sp.log10(theoreticalPvals-betaDown)
# upper = -sp.log10(theoreticalPvals+betaUp)
# pl.fill_between(-sp.log10(theoreticalPvals),lower,upper,color="grey",alpha=0.5)
def addqqplotinfo(qnull,M,xl='-log10(P) observed',yl='-log10(P) expected',xlim=None,ylim=None,alphalevel=0.05,legendlist=None,fixaxes=False):    
    distr='log10'
    pl.plot([0,qnull.max()], [0,qnull.max()],'k')
    pl.ylabel(xl)
    pl.xlabel(yl)
    if xlim is not None:
        pl.xlim(xlim)
    if ylim is not None:
        pl.ylim(ylim)        
    if alphalevel is not None:
        if distr == 'log10':
            betaUp, betaDown, theoreticalPvals = _qqplot_bar(M=M,alphalevel=alphalevel,distr=distr)
            lower = -sp.log10(theoreticalPvals-betaDown)
            upper = -sp.log10(theoreticalPvals+betaUp)
            pl.fill_between(-sp.log10(theoreticalPvals),lower,upper,color="grey",alpha=0.5)
            pl.plot(-sp.log10(theoreticalPvals),lower,'g-.')
            pl.plot(-sp.log10(theoreticalPvals),upper,'g-.')
    if legendlist is not None:
        leg = pl.legend(legendlist, loc=4, numpoints=1)
        # set the markersize for the legend
        for lo in leg.legendHandles:
            lo.set_markersize(10)

    if fixaxes:
        fix_axes()     

#changed xrange to range
def _qqplot_bar(M=1000000, alphalevel = 0.05,distr = 'log10'):
    '''
    calculate error bars for a QQ-plot
    --------------------------------------------------------------------
    Input:
    -------------   ----------------------------------------------------
    M               number of points to compute error bars
    alphalevel      significance level for the error bars (default 0.05)
    distr           space in which the error bars are implemented
                    Note only log10 is implemented (default 'log10')
    --------------------------------------------------------------------
    Returns:
    -------------   ----------------------------------------------------
    betaUp          upper error bars
    betaDown        lower error bars
    theoreticalPvals    theoretical P-values under uniform
    --------------------------------------------------------------------
    '''


    #assumes 'log10'

    mRange=10**(sp.arange(sp.log10(0.5),sp.log10(M-0.5)+0.1,0.1));#should be exp or 10**?
    numPts=len(mRange);
    betaalphaLevel=sp.zeros(numPts);#down in the plot
    betaOneMinusalphaLevel=sp.zeros(numPts);#up in the plot
    betaInvHalf=sp.zeros(numPts);
    for n in range(numPts):
        m=mRange[n]; #numplessThanThresh=m;
        betaInvHalf[n]=st.beta.ppf(0.5,m,M-m);
        betaalphaLevel[n]=st.beta.ppf(alphalevel,m,M-m);
        betaOneMinusalphaLevel[n]=st.beta.ppf(1-alphalevel,m,M-m);
        pass
    betaDown=betaInvHalf-betaalphaLevel;
    betaUp=betaOneMinusalphaLevel-betaInvHalf;

    theoreticalPvals=mRange/M;
    return betaUp, betaDown, theoreticalPvals

#changed nothing
def fix_axes(buffer=0.1):
    '''
    Makes x and y max the same, and the lower limits 0.
    '''    
    maxlim=max(pl.xlim()[1],pl.ylim()[1])    
    pl.xlim([0-buffer,maxlim+buffer])
    pl.ylim([0-buffer,maxlim+buffer])

#changed xrange to range and commented out print(line for lambda)
#took out if addlambda: stuff
def fastlmm_qqplot(pvals, fileout = None, alphalevel = 0.05,legend=None,xlim=None,ylim=None,fixaxes=False,addlambda=False,minpval=1e-20,title=None,h1=None,figsize=[5,5],grid=True):
    '''
    performs a P-value QQ-plot in -log10(P-value) space
    -----------------------------------------------------------------------
    Args:
        pvals       P-values, for multiple methods this should be a list (each element will be flattened)
        fileout    if specified, the plot will be saved to the file (optional)
        alphalevel  significance level for the error bars (default 0.05)
                    if None: no error bars are plotted
        legend      legend string. For multiple methods this should be a list
        xlim        X-axis limits for the QQ-plot (unit: -log10)
        ylim        Y-axis limits for the QQ-plot (unit: -log10)
        fixaxes    Makes xlim=0, and ylim=max of the two ylimits, so that plot is square
        addlambda   Compute and add genomic control to the plot, bool
        title       plot title, string (default: empty)
        h1          figure handle (default None)
        figsize     size of the figure. (default: [5,5])
        grid        boolean: use a grid? (default: True)
    Returns:   fighandle, qnull, qemp
    -----------------------------------------------------------------------
    '''    
    distr = 'log10'
    import pylab as pl
    pl.figure()
    if type(pvals)==list:
        pvallist=pvals
    else:
        pvallist = [pvals]
    if type(legend)==list:
        legendlist=legend
    else:
        legendlist = [legend]
    
    if h1 is None:
        h1=pl.figure(figsize=figsize) 
    
    pl.grid(b=grid, alpha = 0.5)
         
    maxval = 0

    for i in range(len(pvallist)):        
        pval =pvallist[i]
        M = pval.shape[0]
        pnull = (0.5 + sp.arange(M))/M
        # pnull = np.sort(np.random.uniform(size = tests))
                
        pval[pval<minpval]=minpval
        pval[pval>=1]=1

        if distr == 'chi2':
            qnull = st.chi2.isf(pnull, 1)
            qemp = (st.chi2.isf(sp.sort(pval),1))
            xl = 'LOD scores'
            yl = '$\chi^2$ quantiles'

        if distr == 'log10':
            qnull = -sp.log10(pnull)            
            qemp = -sp.log10(sp.sort(pval)) #sorts the object, returns nothing

            print(list(qnull))
            print(list(qemp))

            xl = '-log10(P) observed'
            yl = '-log10(P) expected'
        if not (sp.isreal(qemp)).all(): raise Exception("imaginary qemp found")
        # if qnull.max()>maxval:
            # maxval = qnull.max()                
        pl.plot(qnull, qemp, '.', markersize=2)
        pl.plot([0,qemp.max()], [0,qemp.max()],'r')        
        if addlambda:
            lambda_gc = estimate_lambda(pval)
            # print("lambda=%1.4f" % lambda_gc)
            pl.legend(["gc="+ '%1.3f' % lambda_gc],loc=2)   
            # if there's only one method, just print(the lambda)
            if len(pvallist) == 1:
                legendlist=["$\lambda_{GC}=$%1.4f" % lambda_gc]   
            # otherwise add it at the end of the name
            else:
                legendlist[i] = legendlist[i] + " ($\lambda_{GC}=$%1.4f)" % lambda_gc

    addqqplotinfo(qnull,M,xl,yl,xlim,ylim,alphalevel,legendlist,fixaxes)  
    
    if title is not None:
        pl.title(title)            
    
    if fileout is not None:
        pl.savefig(fileout)

    pl.title("Comparing distributions on log scale")
    # pl.show()
    pl.savefig()
    return h1,qnull, qemp, pl


# ************************************************************************#

#Load the feature matrix
def load_X(file_name_in = 'gwas_X.txt'):
    X=[]
    with open(file_name_in, 'r') as f_in:
        f_in.readline()
        for line in f_in:
            X.append([int(x) for x in list(line.split()[1])])


    X = np.asarray(X)
    X = X.T

    return X

#Load the targets
def load_y(file_name_in = 'gwas_y.txt'):
    y=[]
    with open(file_name_in, 'r') as f_in:
        f_in.readline()
        for line in f_in:
            y.append(int(line.split()[2]))
    y = np.asarray(y)

    return y

#Function for finding delta for the LMM. See section 1.4 in https://media.nature.com/original/nature-assets/nmeth/journal/v8/n10/extref/nmeth.1681-S1.pdf
def find_delta(X, y):

    K = X.dot(X.T)

    S, U = np.linalg.eig(K)
    S=np.real(S)
    U=np.real(U)

    n = len(X)
    UX = U.T.dot(ones)
    Uy = U.T.dot(y)

    def f(log_delta):
        delta = np.exp(log_delta)
        sigma = S + delta * np.identity(S.shape[0])
        # print(sigma)
        sigma_det = np.linalg.det(sigma)
        sigma_inverse = np.linalg.inv(sigma)


        A = UX.T.dot(sigma_inverse.dot(UX))
        b = UX.T.dot(sigma_inverse.dot(Uy))
        w = np.linalg.solve(A,b)

        LL =n*math.log(2*math.pi)
        print(delta)
        # print(S)
        LL += sum([math.log(S[i] + delta) for i in range(S.shape[0])])
        LL += n
        LL += n*math.log(1/n)*sum([math.pow((Uy[i]-UX[i].dot(w)),2)/(S[i]+delta) for i in range(S.shape[0])])
        LL = -1/2*LL
        return LL

    min_results = minimize1D(f=f, minval=-0.5, maxval=1)
    print(min_results)

#Computes the multivariate log likelihood for LMM
def compute_multi(X, y, U, sigma_det, sigma_inverse):
    n = len(X)
    # X_temp = np.hstack((X[:,i].reshape(-1, 1), ones))
    UX = U.T.dot(X)
    Uy = U.T.dot(y)

    A = UX.T.dot(sigma_inverse.dot(UX))
    b = UX.T.dot(sigma_inverse.dot(Uy))
    w = np.linalg.solve(A,b)
    UXw = UX.dot(w)
    sigmag2 = np.asscalar(1/n*(Uy-UXw).T.dot(sigma_inverse.dot(Uy-UXw)))
    LL = np.asscalar(-1/2*(n*math.log(2*math.pi*sigmag2)+math.log(sigma_det)+1/sigmag2*(Uy-UXw).T.dot(sigma_inverse.dot(Uy-UXw))))
    # print(LL)

    return LL

#Runs the LMM on the data
def lmm(X,y):

    K = X.dot(X.T)

    S, U = np.linalg.eigh(K)
    S=np.real(S)
    U=np.real(U)

    delta = 1.000
    sigma = S + delta * np.identity(S.shape[0])
    sigma_det = np.linalg.det(sigma)
    sigma_inverse = np.linalg.inv(sigma)

    p_values = []

    null_ll = compute_multi(ones, y, U, sigma_det, sigma_inverse)

    for i in range(X.shape[1]):
        X_temp = np.hstack((X[:,i].reshape(-1, 1), ones))
        snp_ll = compute_multi(X_temp, y, U, sigma_det, sigma_inverse)
        # print(snp_ll)
        pvalue = scipy.stats.chi2.sf(2*(snp_ll-null_ll),1.0)
        p_values.append(pvalue)
    
    qqplot(np.asarray(p_values), legend="LMM")

#Computes the log likelihood for a given feature matrix and target vector.
def compute_ll(X, y):
    n = len(X)

    XX = X.T.dot(X)
    Xy = X.T.dot(y)
    
    w = np.linalg.solve(XX,Xy)
    Xw = X.dot(w)
    sigma2 = np.asscalar(1/n*(y-Xw).T.dot(y-Xw))
    LL = np.asscalar(-n/2*math.log(2*math.pi*sigma2)-1/(2*sigma2)*(y-Xw).T.dot(y-Xw))

    return LL

#Finds all p-values for null model and for each snp and subjects. P-values are computed as from the chi-squared distribution where the 2*difference in log likelihood is chi-squared with one degree of freedom.
def run_model(X,y, covariates,legend=None):
    p_values = []
    null_ll = compute_ll(covariates, y)

    for i in range(X.shape[1]):
        X_temp = np.hstack((X[:,i].reshape(-1, 1), covariates))
        snp_ll = compute_ll(X_temp,y)
        pvalue = scipy.stats.chi2.sf(2*(snp_ll-null_ll),1.0)
        # neg_log_pvalue = pvalue
        p_values.append(pvalue)

    p_values = np.asarray(p_values)

    # fastlmm_qqplot(np.asarray(p_values), legend=legend)
    return p_values

#Runs the naive model. The ones vector is the "null model".
def naive_model(X,y):
    return run_model(X,y, ones, legend="Naive")

#Gets the first k singular vectors of X
def get_projection(X,k):
    _, _, V = np.linalg.svd(X)    
    V = V.T
    return V[:, :k]

#Adds the PCA projections into the feature matrix. The PCA projections and the ones vector consistute the "null model".
def pca_corrected_model(X,y):

    P = get_projection(X, 3)
    X_proj = X.dot(P)

    # pickle.dump(X_proj, open('X_proj.p', 'wb'))
    # X_proj = pickle.load(open('X_proj.p','rb'))

    covariates = np.hstack((X_proj, ones))
    #X + X_proj, figure out what X_proj?, don't give the above lines
    return run_model(X,y, covariates, "PCA-corrected")

#Runs LMM. p-values were taken from results done with FastLMM.
def lmm():
    lmm_pvals = np.asarray([float(line.strip()) for line in open('lmm.pvals.txt', 'r')])
    # fastlmm_qqplot(lmm_pvals, legend='LMM')
    return lmm_pvals

#Load the data
# X = load_X()
# y=load_y()

# #Normalize the columns
# X = preprocessing.normalize(X, norm='l2', axis = 0)
# y = y.reshape(-1, 1)

# ones = np.ones(y.shape)
# ones=ones.reshape(-1, 1)

# naive_pvals= naive_model(X,y)
# pca_pvals = pca_corrected_model(X,y)
# lmm_pvals = lmm()




