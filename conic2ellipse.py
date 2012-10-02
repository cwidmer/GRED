#!/usr/bin/env python2.5
#
# Written (W) 2011 Christian Widmer
# Copyright (C) 2011 Max-Planck-Society

"""
@author: Christian Widmer

@summary: module to convert conic parameterizations to 

"""

import numpy
import math


def conic_to_rotated_ellipse(A, B, C, D, E, F):
    """
    the most general case for the ellipse, including translation and rotation
    """

    #%Assume a=1.
    #a=1;
    #[b c d E g]=deal(X(1), X(2), X(3), X(4), X(5));

    #%discriminant matrix
    #J=[a b;b c];
    J = numpy.vstack([numpy.hstack([A,B]),
                      numpy.hstack([C,D])])

    #%quadratic curve matrix
    #Q=[a b d;b c E;d E g];
    #Q = numpy.array([[A, B, D], [B, C, E], [D, E, F]])
    Q = numpy.vstack([numpy.hstack([A, B, D]),
                      numpy.hstack([B, C, E]),
                      numpy.hstack([D, E, F])])
    print J.shape, Q.shape

    # s=sqrt((a-c)^2+4*b^2);
    s = numpy.sqrt((A-C)**2 + 4*B**2)

    #%center oE the ellipse
    #xc = (c*d-b*E)/(.linalg.det(J));
    #yc=(a*E-b*d)/(.linalg.det(J));
    dj = numpy.linalg.det(J)
    dq = numpy.linalg.det(Q)
    xc = (C*D-B*E) / (-dj)
    yc = (A*E-B*D) / (-dj)

    #%semi-major and semi-minor axes
    #a_prime=sqrt(2.linalg.det(Q)/.linalg.det(J)*(s-(a+c))));
    #b_prime=sqrt(2.linalg.det(Q)/.linalg.det(J)*(-s-(a+c))));

    a_prime = numpy.sqrt(2*dq / (dj*(s-(A+C))))
    b_prime = numpy.sqrt(2*dq / (dj*(-s-(A+C))))

    #%tilt angle phi
    #phi=0.5*(atan(2*b/c-a));
    phi = 0.5*(math.atan(2*B/C-A))

    semimajor = max(a_prime, b_prime)
    semiminor = min(a_prime, b_prime)

    if (a_prime < b_prime):
        phi = math.pi/2 - phi

    return xc, yc, a_prime, b_prime, phi



def derive_conic_to_aligned_ellipse():
    """
    derive the respective routine using sympy
    """

    import sympy
    cx = sympy.Symbol("cx")
    cy = sympy.Symbol("cy")
    rx = sympy.Symbol("rx")
    ry = sympy.Symbol("ry")
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    fun = ((x-cx)**2 / (rx**2) + (y-cy)**2 / (ry**2) - 1)**2

    print "initial function"
    sympy.pprint(fun)

    f1 = fun.expand(deep=True)

    f2 = sympy.together(f1)

    f3 = f2*(rx**2*ry**2)

    print "======================="
    print "expanded function"
    sympy.pprint(f3)

    print "======================="
    print "derive gradient"
    print f3.diff(cx)
    print f3.diff(cy)
    print f3.diff(rx)
    print f3.diff(ry)

    return f3


def derive_gradient_ellipse_squared():
    """
    derive the gradient for ellipse with squared loss
    """

    import sympy
    cx = sympy.Symbol("cx")
    cy = sympy.Symbol("cy")
    rx = sympy.Symbol("rx")
    ry = sympy.Symbol("ry")
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")

    fun = ((x-cx)**2 / (rx**2) + (y-cy)**2 / (ry**2) - 1)**2

    print fun.diff(cx)
    print fun.diff(cy)
    print fun.diff(rx)
    print fun.diff(ry)


    # example for smoothness regularizer
    r1 = sympy.Symbol("r1")
    r2 = sympy.Symbol("r2")
    r3 = sympy.Symbol("r3")
    r4 = sympy.Symbol("r4")

    reg = (r1 - r2)**2 + (r2 - r3)**2 + (r3 - r4)**2
    print reg.diff(r1)
    print reg.diff(r2)
    print reg.diff(r3)
    print reg.diff(r4)



def conic_to_aligned_ellipse(A, B, D, E, F):
    """
    the case where the ellipse is aligned with the axes
    """

    ry = numpy.sqrt(A)
    rx = numpy.sqrt(B)

    cx = - D / (2*A)
    cy = - E / (2*B)

    F_new = cx**2 * rx**2 + cy**2 * rx**2 - rx**2 * ry**2
    print F, F_new

    return cx, cy, rx, ry



def conic2ellipsoid(A, bv, c):
    '''
    function [z, a, b, alpha] = conic2parametric(A, bv, c)
    '''
    from math import *
    from numpy import *

    ## Diagonalise A - find Q, D such at A = Q' * D * Q
    D, Q = linalg.eig(A)
    Q = Q.T
    
    ## If the determinant < 0, it's not an ellipse
    if prod(D) <= 0:
        raise RuntimeError, 'fitellipse:NotEllipse Linear fit did not produce an ellipse'
    
    ## We have b_h' = 2 * t' * A + b'
    t = -0.5 * linalg.solve(A, bv)
    
    c_h = dot( dot( t.T, A ), t ) + dot( bv.T, t ) + c
    
    z = t
    rx = sqrt(-c_h / D[0])
    ry = sqrt(-c_h / D[1])
    rz = sqrt(-c_h / D[2])

    alpha = atan2(Q[0,1], Q[0,0])
    
    return z, rx, ry, rz, alpha


if __name__ == "__main__":

    derive_gradient_ellipse_squared()

