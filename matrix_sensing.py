
import tensorflow as tf
import numpy as np

import cvxpy as cp

import os
#avoids some annoying warning message for me
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Some friendly values for testing purposes
U_start = np.array([[0.01,0.05,0.06],
                    [0.09,0.04,0.03],
                    [0.08,0.02,0.07]])
As = np.array([[[1.0,0.0,0.0],
                [0.0,0.0,0.0],
                [0.0,0.0,0.0]],
               [[0.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,0.0]],
               [[0.0,0.0,0.0],
                [0.0,0.0,1.0],
                [0.0,1.0,0.0]]])
ys = np.array([1.0, 2.0, -1.0])

# try: U_new = GradientFlow(U_start, (As, ys), 10000, learningRate=1e-4)

def GradientFlow(U_start, As, ys, numSteps, learningRate=1e-4,verbose=False):
    As, ys = tf.constant(As), tf.constant(ys)
    U = tf.Variable(U_start)
    for i in range(numSteps):
        DoGradientStep(U, As, ys, learningRate)
        if verbose:
            #print("hi")
    return U

@tf.function
def DoGradientStep(U, As, ys, learningRate):
    dydU = Gradient(U, As, ys)
    U.assign_sub(learningRate*dydU)

@tf.function
def Gradient(U, A, ys):
    """ The goal is that <As[i], U @ U.T> = ys[i] for each index i. """
    with tf.GradientTape() as g:
        g.watch(U)
        y = ParametrizedSquaredError(U, As, ys)
    return g.gradient(y, U)

@tf.function
def ParametrizedSquaredError(U, As, ys):
    return SquaredError(U @ tf.transpose(U), As, ys)

def SquaredError(X, As, ys):
    """ This is the quantity that the gradient flow tries to minimize. """
    As_T = tf.transpose(As, perm=[0,2,1])
    innerProducts = tf.linalg.trace(As_T @ X)
    errors = ys - innerProducts
    return tf.reduce_sum(errors**2)

#################

def MinimizeNuclearNorm(U_start, As, ys):
    raise NotImplementedError

#################

def ExactMinNuclearNorm(As,ys):
    n = np.shape(As[0])[0]
    return SimpleSdp(As,ys,np.eye(n))


################

def SimpleSdp(A, b, C):
    
    n = np.shape(C)[0]
    
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    for i in range(len(A)):
        constraints += [cp.trace(A[i] @ X) == b[i]]

    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                  constraints)

    prob.solve()
    
    return prob.value


####################



def testSquaredError():
    X = np.array([[1.0,0.0],
                  [1.0,-1.0]])
    As = np.array([[[1.0,0.0],
                    [0.0,1.0]],
                   [[0.0,1.0],
                    [2.0,0.0]],
                   [[0.0,0.0],
                    [0.0,3.0]]])
    ys = np.array([0.0, 2.0, -3.0])
    observations = (As, ys)
    assert SquaredError(X, observations) < 1e-9
print("hello")

###################



U = GradientFlow(U_start, As, ys, 200, learningRate=1e-4)
print(U)
