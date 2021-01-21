
from hyperbolic_turtle import Turtle, Point, tau, distance
from scipy.optimize import minimize
from collections import namedtuple


    
def demo():
    P,Q,R = equilateralTriangle(1)
    X = (0,0)
    fP, fQ, fR = (P,0), (Q,1), (R,1) # given function points
    print(f"Trying to get an upper bound at {X}...")
    point, bound = tryUpperBound([[fP,fQ],fR], X)
    print(f"Simple tree: upper bound {bound} at {point}")
    point, bound = tryUpperBound([[fP,fR],fQ], X)
    print(f"Simple tree: upper bound {bound} at {point}")
    point, bound = tryUpperBound([[fR,fQ],fP], X)
    print(f"Simple tree: upper bound {bound} at {point}")
    point, bound = tryUpperBound([[[[fR,fQ],fP],[[fP,fQ],fR]],
                                  [[[fR,fQ],fP],[[fP,fR],fQ]]], X)
    print(f"Complicated tree: upper bound {bound} at {point}")



def tryUpperBound(pointTree, X):
    def score(thetas):
        point, bound = _treeUpperBound(pointTree, thetas)
        return bound
    def shouldBeZero1(thetas):
        point, bound = _treeUpperBound(pointTree, thetas)
        return point[0] - X[0]
    def shouldBeZero2(thetas):
        point, bound = _treeUpperBound(pointTree, thetas)
        return point[1] - X[1]
    thetas = minimize(
        fun=score,
        x0=[0.5 for i in range(_numThetas(pointTree))],
        bounds=[(0,1) for i in range(_numThetas(pointTree))],
        constraints=[{"fun":shouldBeZero1, "type":"eq"},
                     {"fun":shouldBeZero2, "type":"eq"}]
        ).x
    return _treeUpperBound(pointTree, thetas)

### manipulating trees of points ###

def _treeUpperBound(pointTree, thetas):
    """
    Input is something like
        pointTree=[[[fA,fB],fC],[fA,fD]], thetas=[0.5, 0.2, 0.1, 0.4]
    returns (point, upper bound at that point). """
    if _numThetas(pointTree) == 0:
        return pointTree # base case
    else:
        upperBound0 = _treeUpperBound(pointTree[0],
                                      thetas[:_numThetas(pointTree[0])])
        upperBound1 = _treeUpperBound(pointTree[1],
                                      thetas[_numThetas(pointTree[0]):-1])
        return weightedAverage(upperBound0, upperBound1, thetas[-1])

def _numThetas(pointTree):
    return (0 if (len(pointTree)==2 and type(pointTree[1]) in (int,float))
              else 1 + _numThetas(pointTree[0]) + _numThetas(pointTree[1]))
    
def weightedAverage(fP, fQ, theta):
    """ fP is (hyperbolic point P, number f(P)).
        returns the weighted average fP*theta + fQ*(1-theta). """
    P, f_P = fP
    Q, f_Q = fQ
    functionValue = theta*f_P + (1-theta)*f_Q
    if P == Q: return (P, functionValue)
    T = Turtle(P, Q)
    T2 = T.forward((1-theta)*distance(P, Q))
    return (T2.point, functionValue)
 
### utils ###

def equilateralTriangle(r):
    """ Returns vertices of an equilateral triangle centered at the origin. """
    T = Turtle((0,0),(0.1,0))
    turtles = [T, T.right(tau/3), T.right(2*tau/3)]
    turtles = [t.forward(r) for t in turtles]
    return [t.point for t in turtles]
