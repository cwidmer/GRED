#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Christian Widmer
# Copyright (C) 2011 Max-Planck-Society

"""
@author: Christian Widmer
@summary: Code for ellipsoid

"""

import scipy.optimize
import numpy
import random
import loss_functions
#import cvxmod
#from cvxmod.atoms import square
#import kdtree_cook as kdt
import scipy.spatial.distance as dist

from kdtree import KDTree
from util import ellipsoid
from data_processing import get_data_example

# global variables to use inside of solver
x = [0.0, -1.0, 1.0,  0.0, 0.0, 0.0]
y = [1.0,  0.0, 0.0, -1.0, 0.0, 0.0]
z = [0.0,  0.0, 0.0,  0.0, 1.0,-1.0]
i = [0.0,  0.0, 0.0,  0.0, 1.0,-1.0]



def fitting_obj_sample(param):
    """
    computes residuals based on distance from ellipsoid
    
    can be used with different loss-functions on residual
    """


    obj = 0

    # centers
    cx = param[0]
    cy = param[1]
    cz = param[2]

    rx = param[3]
    ry = param[4]
    rz = param[5]
    
    sx, sy, sz = ellipsoid(cx, cy, cz, rx, ry, rz, 20)
    num_samples = len(sx)

    #plot_point_cloud(sx, sy, sz)

    print "num_samples", num_samples

    #import pdb    
    #pdb.set_trace()

    #data = numpy.array(zip(sx, sy, sz)).T
    #tree = kdt.kdtree( data, leafsize=1000 )

    data = zip(sx, sy, sz)
    tree = KDTree.construct_from_data(data)

    num_queries = len(x)

    print "num_queries", num_queries

    global global_loss
    global_loss = numpy.zeros(num_queries)

    for idx in range(num_queries):

        """
        Compute the unique root tbar of F(t) on (-e2*e2,+infinity);

        x0 = e0*e0*y0/(tbar + e0*e0);
        x1 = e1*e1*y1/(tbar + e1*e1);
        x2 = e2*e2*y2/(tbar + e2*e2);

        distance = sqrt((x0 - y0)*(x0 - y0) + (x1 - y1)*(x1 - y1) + (x2 - y2)*(x2 - y2))
        """

        query = (x[idx], y[idx], z[idx])
        nearest, = tree.query(query_point=query, t=1)
        residual = dist.euclidean(query, nearest)

        #obj += loss_functions.squared_loss(residual)
        #obj += loss_functions.abs_loss(residual)
        #obj += loss_functions.eps_loss(residual, 2)
        #obj += loss_functions.eps_loss_bounded(residual, 2)
        loss_xt = loss_functions.eps_loss_asym(residual, 2, 1.0, 0.2)
        obj += loss_xt
        global_loss[idx] = num_queries

        #obj += eps_loss(residual, 2)*data_intensity[idx]

    # add regularizer to keep radii close
    reg = 10 * regularizer(param)

    print "loss", obj
    print "reg", reg

    obj += reg

    return obj



def regularizer(param):
    """
    enforces similar radii
    """

    rx = param[3]
    ry = param[4]
    rz = param[5]

    reg = (rx - ry)**2 + (rx - rz)**2 + (ry - rz)**2

    return reg



def fitting_obj(param):
    """
    computes residuals based on distance from ellipsoid
    
    can be used with different loss-functions on residual
    """


    # centers
    cx = param[0]
    cy = param[1]
    cz = param[2]

    radius = param[3]

    #a = param[3]
    #b = param[4]
    #c = param[5]

    obj = 0

    for idx in range(len(x)):

        """
        Compute the unique root tbar of F(t) on (-e2*e2,+infinity);

        x0 = e0*e0*y0/(tbar + e0*e0);
        x1 = e1*e1*y1/(tbar + e1*e1);
        x2 = e2*e2*y2/(tbar + e2*e2);

        distance = sqrt((x0 - y0)*(x0 - y0) + (x1 - y1)*(x1 - y1) + (x2 - y2)*(x2 - y2))
        """

        #residual =  b*b*c*c*(cx - x[idx])**2 
        #residual += a*a*c*c*(cy - y[idx])**2 
        #residual += a*a*b*b*(cz - z[idx])**2
        #residual = residual - a*a*b*b*c*c

        residual =  (cx - x[idx])**2 + (cy - y[idx])**2 + (cz - z[idx])**2 
        residual = numpy.sqrt(residual) - radius

        tmp = loss_functions.squared_loss(residual)
        #tmp = loss_functions.abs_loss(residual)
        #tmp = loss_functions.eps_loss(residual, 1)
        #tmp = loss_functions.eps_loss_asym(residual, 2, 1.0, 0.3)

        # consider intensity
        obj += tmp*i[idx]


    return obj



def fit_ellipsoid(dx, dy, dz, di, num_points=None):
    """
    fit ellipoid beased on data
    """


    is_sphere = True

    num_points = len(dx)
    idx = range(num_points)
    random.shuffle(idx)    
    subset_idx = idx[0:500]

    global x,y,z,i
    x = numpy.array(dx)[subset_idx]
    y = numpy.array(dy)[subset_idx]
    z = numpy.array(dz)[subset_idx]
    i = numpy.array(di)[subset_idx]
     
    print "num data points: %i" % (len(x))

    if is_sphere:
        x0 = numpy.array([0, 0, 0, 5])
    else:
        x0 = numpy.array([15, 15, 10, 5, 5, 5])
    x0[0] = numpy.average(x)
    x0[1] = numpy.average(y)
    x0[2] = numpy.average(z)

    print "center guess: x=%f, y=%f, z=%f" % (x0[0], x0[1], x0[2])

    #x_opt = scipy.optimize.fmin(fitting_obj, x0)
    epsilon = 0.5

    bounds = []
    bounds.append((0, None)) # cx
    bounds.append((0, None)) # cy
    bounds.append((0, None)) # cz
    bounds.append((0, None)) # rx

    if not is_sphere:
        bounds.append((0, None)) # ry
        bounds.append((0, None)) # rz

    if is_sphere:
        #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
        #x_opt = scipy.optimize.fmin(fitting_obj_sphere_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
        #x_opt = scipy.optimize.fmin(fitting_obj, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
        x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj, x0, bounds=bounds, approx_grad=True, messages=5)
        return x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[3], x_opt[3]

    else:
        #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
        x_opt = scipy.optimize.fmin(fitting_obj_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
        return x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5]



def fit_ellipsoid_cvx(x, y, z):
    """
    fit ellipoid using squared loss

    """

    #TODO not working. it is using non-linear solver, but takes forever

    assert len(x) == len(y)

    N = len(x)
    D = 7
    
    dat = numpy.zeros((N, D))
    dat[:,0] = x*x
    dat[:,1] = y*y
    dat[:,2] = z*z
    dat[:,3] = x
    dat[:,4] = y
    dat[:,5] = z
    dat[:,6] = numpy.ones(N)
    

    print dat.shape   
    dat = cvxmod.matrix(dat)
    #### parameters

    # data
    X = cvxmod.param("X", N, D)


    #### varibales
    
    # parameter vector
    theta = cvxmod.optvar("theta", D)
    

    # simple objective 
    objective = cvxmod.sum(square(X*theta))
    
    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    p.constr.append(theta[0] + theta[1] == 1)
    #p.constr.append(theta[0] + theta[2] == 1)
    #p.constr.append(theta[1] + theta[2] == 1)
    
    
    ###### set values
    X.value = dat
    #solver = "mosek" 
    #p.solve(lpsolver=solver)
    p.solve()
    

    w = numpy.array(cvxmod.value(theta))
    
    #print weights
    
    cvxmod.printval(theta)


    ## For clarity, fill in the quadratic form variables
    A              = numpy.zeros((3,3))
    A[0,0]         = w[0]
    #A.ravel()[1:3] = 0 #w[2]
    A[1,1]         = w[1]
    A[2,2]         = w[2]
    bv             = w[3:6]
    c              = w[6]
    
    ## find parameters
    from conic2ellipse import conic2ellipsoid
    z, rx, ry, rz, alpha = conic2ellipsoid(A, bv, c)

    return z, rx, ry, alpha


if __name__ == "__main__":

    # debug
    dx, dy, dz, di = get_data_example()
    
    #print fit_stack(dx, dy, dz, di)
    print fit_ellipsoid_cvx(dx, dy, dz)

