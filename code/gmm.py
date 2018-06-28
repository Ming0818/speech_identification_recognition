from sklearn.model_selection import train_test_split
from scipy.misc import logsumexp
import numpy as np
import os, fnmatch
import random
import math

dataDir = '/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))

# helper to computer preComputedForM
def preComputedForM(myTheta, M = 8, D = 13):

    result = []
    for m in range(M):
        log_denom = ((D/2)*math.log((math.pi*2)))
        prod = np.prod(myTheta.Sigma[m])
        if prod == 0:
            log_denom += 0.5*math.log(1)
        else:
            log_denom += 0.5*math.log(prod)

        result.append(log_denom)

    return result

# helper to computer log_b for X instead of x
def log_b_m_X( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''

    log_num = np.sum(np.divide(np.square(np.subtract(X,myTheta.mu[m])),myTheta.Sigma[m]), axis=1)
    answer = np.subtract(np.multiply(log_num,-0.5),preComputedForM[m])
    return answer


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''

    D = myTheta.mu.shape[1]
    if preComputedForM == []:
        log_num = (-0.5 * np.sum([((x[d] - myTheta.mu[m][d])**2)/myTheta.Sigma[m][d] for d in range(D)]))
        log_denom = math.log(((math.pi*2)**(D/2))*(np.prod(myTheta.Sigma[m]) ** 0.5))
        answer = log_num - log_denom
        return answer
    else:
        log_num = (-0.5 * np.sum([((x[d] - myTheta.mu[m][d])**2)/myTheta.Sigma[m][d] for d in range(D)]))
        answer = log_num - preComputedForM[m]
        return answer

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''

    M = myTheta.omega.shape[0]
    log_wmbm = math.log(myTheta.omega[m]) + log_b_m_x(m, x, myTheta)
    sum_log_wkbk =  logsumexp([math.log(myTheta.omega[k]) + log_b_m_x(k, x, myTheta) for k in range(M)])
    answer = log_wmbm - sum_log_wkbk
    return answer

def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    M = myTheta.omega.shape[0]

    answer = np.sum(logsumexp(np.array([np.add(math.log(myTheta.omega[m]), log_Bs[m]) for m in range(M)]), axis=0))

    return answer

def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    # initialize
    D = X.shape[1]
    myTheta = theta( speaker, M, D )
    myTheta.omega = np.ones((M,1))
    for i in range(M):
        rand = random.randint(0,D)
        myTheta.mu[i] = X[rand]
        myTheta.Sigma[i] = np.ones((1,D))

    T = X.shape[0]
    iter = 0
    prev_L = float('-inf')
    improvement = float('inf')
    while iter <= maxIter and improvement >= epsilon:

        log_b = np.ones((M,T))
        log_p = np.ones((M,T))

        preCompM = preComputedForM(myTheta, M, X.shape[1])

        # use log_b_m_X instead of log_b_m_x to increase speed.
        # calculate both log_p and log_b without using log_b_m_x or log_p_m_x
        for m in range(M):
            # log_b_m_X
            log_b[m] = log_b_m_X(m, X, myTheta, preCompM)

            # log_p_m_X
            log_wmbm = np.add(math.log(myTheta.omega[m]), log_b_m_X(m, X, myTheta, preCompM))
            sum_log_wkbk = logsumexp(np.array([np.add(math.log(myTheta.omega[k]), log_b_m_X(k, X, myTheta, preCompM)) for k in range(M)]), axis=0)
            log_p[m] = np.subtract(log_wmbm,sum_log_wkbk)

        # for m in range(M):
        #     for t in range(T):
        #         log_b[m][t] = log_b_m_x(m, X, myTheta, preCompM)
        #         log_p[m][t] = log_p_m_x(m, X[t], myTheta)

        # computer log likelihood
        L = logLik(log_b, myTheta)

        # update Theta
        for m in range(M):
            sum_p = np.sum(np.exp(log_p[m]))
            myTheta.omega[m] = sum_p/T

            X_trans = np.transpose(X)
            p_X = np.transpose(np.multiply(np.exp(log_p[m]), X_trans))
            myTheta.mu[m] = np.divide(np.sum(p_X, axis=0),sum_p)

            p_X2 = np.multiply(np.exp(log_p[m]), np.transpose(np.square(X)))
            sigma = np.divide(np.sum(np.transpose(p_X2), axis=0), sum_p)
            myTheta.Sigma[m] = np.subtract(sigma, np.square(myTheta.mu[m]))

        improvement = L - prev_L
        prev_L = L
        iter = iter + 1

    return myTheta

def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    M = models[0].omega.shape[0]
    T = mfcc.shape[0]
    Logs = {}

    for i in range(len(models)):
        log_Bs = np.ones((M, T))
        preCompM = preComputedForM(models[i], M, mfcc.shape[1])

        # for m in range(M):
        #     log_Bs[m] = log_b_m_X(m, mfcc, models[i], preCompM)

        for m in range(M):
            for t in range(T):
                log_Bs[m][t] = log_b_m_x(m, mfcc[t], models[i], preCompM)

        L = logLik(log_Bs, models[i])
        speaker = i
        Logs[speaker] = L

    sorted_Logs = sorted(((value,key) for (key,value) in Logs.items()), reverse=True)

    bestModel = sorted_Logs[0][1]

    print(models[correctID].name)
    j = 0
    while j < 5:
        print(models[sorted_Logs[j][1]].name, sorted_Logs[j][0])
        j += 1

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )

            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k )
    accuracy = 1.0*numCorrect/len(testMFCCs)
