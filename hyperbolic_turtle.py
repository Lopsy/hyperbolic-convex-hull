
"""
Try:

T = Turtle((0,0), (0.1,0)) # Makes a turtle at the origin pointing to the right
(functionally the same as Turtle((0,0), (x,0)) for any positive x < 1)
T2 = T.forward(3)
T3 = T2.left(pi/8) # in radians
T4 = T3.face((0,0))

"""

import math
from math import tanh, tau
from random import random

from hyperbolic.poincare.shapes import *
from hyperbolic.poincare import Transform
import hyperbolic.tiles as htiles

class Turtle:
    def __init__(self, point, head):
        """ A turtle is centered at `point` and points towards `head`. """
        self.point, self.head = point, head
        assert len(point) == len(head) == 2
        assert 0 <= sum(x**2 for x in point) < 1# must be inside Poincare disc
        assert 0 <= sum(x**2 for x in head) < 1 # must be inside Poincare disc
    def transformToOrigin(self):
        """ Transformation mapping self to origin, pointing along x-axis. """
        return Transform.shiftOrigin(self.point, self.head)
    def forward(self, distance):
        TForward = Transform.shiftOrigin(Point(math.tanh(-distance/2), 0))
        TToOrigin = self.transformToOrigin()
        TFromOrigin = TToOrigin.inverted()
        T = Transform.merge(TToOrigin, TForward, TFromOrigin)
        return Turtle(*T(self.point, self.head))
    def left(self, radians):
        TToOrigin = Transform.shiftOrigin(self.point)
        TRotation = Transform.merge(TToOrigin, Transform.rotation(rad=radians),
                           TToOrigin.inverted())
        return Turtle(self.point, TRotation(self.head)[0])
    def right(self, radians):
        return self.left(-radians)
    def face(self, target):
        # alternate implementation: return Turtle(self.point, target) !
        toOrigin = self.transformToOrigin()
        angleToRotate = toOrigin(target)[0].theta
        return self.left(angleToRotate)
    def __repr__(self):
        return str(self.point)+" toward "+str(self.head)

def distance(P, Q):
    try:
        return Point(*P).distanceTo(Point(*Q))
    except: # some floating point nonsense happened with two very close points
        assert abs(P[0]-Q[0]) < 1e-8
        assert abs(P[1]-Q[1]) < 1e-8
        # so just approximate idk
        euclideanDistance = ((P[0]-Q[0])**2 + (P[1]-Q[1])**2)**0.5
        distanceToOrigin = (P[0]**2+P[1]**2)**0.5
        return euclideanDistance * 2 / (1 - distanceToOrigin**2)

def randomPoint():
    angle = random() * math.tau
    d = min(random(), random())
    return Point(d*math.cos(angle), d*math.sin(angle))

def Translation(P, Q):
    def negative(P): return Point(-P.x, -P.y)
    TPO = Transform.shiftOrigin(P) # sends P to origin
    Treal = Transform.shiftOrigin(negative(TPO(Q)[0])) # sends origin to TPO(Q)
    return Transform.merge(TPO, Treal, TPO.inverted())
def RotationAt(P, radians):
    TPO = Transform.shiftOrigin(P) # sends P to origin
    return Transform.merge(TPO, Transform.rotation(rad=radians),
                           TPO.inverted())
