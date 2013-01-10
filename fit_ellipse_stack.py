#!/usr/bin/env python2.5
#
# Written (W) 2011-2013 Christian Widmer
# Copyright (C) 2011-2013 Max-Planck-Society, TU-Berlin, MSKCC

"""
@author: Christian Widmer
@summary: Provides routines for fitting ellipse stacks in original parameter space

"""


import scipy.optimize
import numpy
import loss_functions
import util
from util import Ellipse
from autowrapped_loss import Loss



def fitting_obj_stack(param, x, y, z, i):
    """
    computes residuals based on distance from ellipsoid
    
    can be used with different loss-functions on residual
    """


    # centers
    cx = param[0]
    cy = param[1]

    #num_layers = len(set(z))
    #assert len(param) == num_layers+2

    radii = param[2:]

    num_layers = len(radii) / 2
    radii_x = radii[0:num_layers]
    radii_y = radii[num_layers:]

    obj = 0


    for idx in range(len(x)):

        rx = radii_x[z[idx]]
        ry = radii_y[z[idx]] 
        residual = (cx - x[idx])**2 / (rx**2) + (cy - y[idx])**2 / (ry**2) - 1

        tmp = loss_functions.squared_loss(residual)
        #obj += loss_functions.eps_loss(residual, 2)*data_intensity[idx]
        #obj += loss_functions.abs_loss(residual)
        #tmp = loss_functions.eps_loss(residual, 1)
        #obj += loss_functions.eps_loss_asym(residual, 2, 1.0, 0.3)
        #print idx, residual

        #obj += tmp*i[idx]
        obj += tmp


    #gradient = gradient_c + gradient_rx + gradient_ry
    # smoothness regularizer
    #for idx in xrange(len(radii_x)-1):
    #    obj += (radii_x[idx] - radii_x[idx+1])**2

    # smoothness regularizer
    #for idx in xrange(len(radii_y)-1):
    #    obj += (radii_y[idx] - radii_y[idx+1])**2

    # L1-regularize large radii
    #for r in radii:
    #    obj += r

    return obj


def fitting_obj_stack_gradient(param, dx, dy, dz, di, loss):
    """
    computes objective and gradient for smoothness-regularized least squares ellipse stack
    """


    # centers
    cx = param[0]
    cy = param[1]

    wx = param[2]
    wy = param[3]

    #num_layers = len(set(z))
    #assert len(param) == num_layers+2

    radii = param[4:]

    num_layers = len(radii) / 2

    #print "num_layers", num_layers

    radii_x = radii[0:num_layers]
    radii_y = radii[num_layers:]

    assert len(radii_x) == len(radii_y)

    obj = 0

    gradient_c = [0.0, 0.0]
    gradient_w = [0.0, 0.0]
    gradient_rx = [0.0]*(len(radii_x))
    gradient_ry = [0.0]*(len(radii_y))

    min_z = min(dz)
    mean_di = float(numpy.mean(di))

    for idx in range(len(dx)):

        x = dx[idx]
        y = dy[idx]
        z = dz[idx]

        # determine correct layer
        z_idx = z - min_z

        rx = radii_x[z_idx]
        ry = radii_y[z_idx]

        # normalized contribution
        i = float(di[idx]) / mean_di

        # loss obj and gradient
        obj += i*loss.get_obj(x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_c[0] += i*loss.get_grad("cx", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_c[1] += i*loss.get_grad("cy", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_w[0] += i*loss.get_grad("wx", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_w[1] += i*loss.get_grad("wy", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_rx[z_idx] += i*loss.get_grad("rx", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_ry[z_idx] += i*loss.get_grad("ry", x, y, z, cx, cy, wx, wy, rx, ry)


    ############################
    # smoothness regularizer using L2-regularization

    reg_smoothness = True

    if reg_smoothness:

        for idx in xrange(num_layers-1):
            
            obj += (radii_x[idx] - radii_x[idx+1])**2
            obj += (radii_y[idx] - radii_y[idx+1])**2

            # compute gradient
            if idx == 0:
                gradient_rx[idx] += 2*radii_x[0] - 2*radii_x[1]
                gradient_ry[idx] += 2*radii_y[0] - 2*radii_y[1]
            else:
                gradient_rx[idx] += 4*radii_x[idx] - 2*radii_x[idx-1] - 2*radii_x[idx+1]
                gradient_ry[idx] += 4*radii_y[idx] - 2*radii_y[idx-1] - 2*radii_y[idx+1]

        # last entry of gradient
        gradient_rx[-1] += 2*radii_x[-1] - 2*radii_x[-2]
        gradient_ry[-1] += 2*radii_y[-1] - 2*radii_y[-2]


    ############################
    # enfore small radii at the ends by means of L1-regularization

    reg_end_param = 10
    num_end_layers = 1

    for idx in xrange(num_end_layers):
        obj += reg_end_param*(radii_x[idx] + radii_x[-idx-1])
        obj += reg_end_param*(radii_y[idx] + radii_y[-idx-1])

        # last entry of gradient
        gradient_rx[idx] += reg_end_param
        gradient_rx[-idx-1] += reg_end_param
        gradient_ry[idx] += reg_end_param
        gradient_ry[-idx-1] += reg_end_param


    # L1-regularize large radii
    #for idx, rx in enumerate(radii_x):
    #    obj += rx
    #    gradient_rx[idx] += 1

    # L1-regularize large radii
    #for idx, ry in enumerate(radii_y):
    #    obj += ry
    #    gradient_ry[idx] += 1

    # L1-regularize large w
    reg_param_w = 5
    obj += wx*reg_param_w
    obj += wy*reg_param_w
    gradient_w[0] += reg_param_w
    gradient_w[1] += reg_param_w


    # build final gradient
    gradient = gradient_c + gradient_w + gradient_rx + gradient_ry

    #return gradient
    return obj, gradient


def check_gradient():
    """
    sanity check for gradient that compares the analytical gradient 
    to one computed numerically by finite differences
    """

    n = 10
    num_z = 10


    x = []
    y = []
    z = []
    i = []

    for idx in range(num_z):
        dat = util.Ellipse(1, 1, 1, 2, 0).sample_uniform(n)
    
        x += list(dat[0])
        y += list(dat[1])
        z += [idx]*(n+1)
        i += [1.0]*(n+1)

    assert len(x) == len(y) == len(z) == len(i)

    x0 = [3.0]*((max(z)+1)*2 + 2 + 2)

    print "len(x0) = %i" % len(x0)

    loss = Loss("algebraic_squared")

    # wrap function
    def func(param, x, y, z, i):
        return fitting_obj_stack_gradient(param, x, y, z, i, loss)[0]

    def func_prime(param, x, y, z, i):
        return fitting_obj_stack_gradient(param, x, y, z, i, loss)[1]


    print scipy.optimize.check_grad(func, func_prime, x0, x, y, z, i)




def fit_ellipse_stack_scipy(dx, dy, dz, di, loss_type = "algebraic_abs"):
    """
    fit ellipoid based on scipy optimize
    """

    #global x,y,z,i
    x = numpy.array(dx)
    y = numpy.array(dy)
    z = numpy.array(dz)
    i = numpy.array(di)

    min_z = min(z)
    num_layers = max(z) - min_z + 1
    print "number of active layers", num_layers
    print "num data points: %i" % (len(x))

    initial_radius = 3.0

    num_parameters = 2 + 2 + num_layers*2 # center + w-vector + radii
    x0 = numpy.ones(num_parameters)*initial_radius
    x0[0] = numpy.average(x)
    x0[1] = numpy.average(y)
    x0[2] = 0
    x0[3] = 0

    #x_opt = scipy.optimize.fmin(fitting_obj, x0)
    epsilon = 0.1

    # contrain all variables to be positive
    bounds = [(0, None) for _ in range(num_parameters)]
    bounds[2] = (None, None) # no positivity for w0
    bounds[3] = (None, None) # no positivity for w1

    assert len(bounds) == len(x0)

    print "fitting ellipse stack with loss", loss_type
    loss = Loss(loss_type)

    #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
    #x_opt = scipy.optimize.fmin(fitting_obj_sphere_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    #x_opt = scipy.optimize.fmin(fitting_obj_stack, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    #x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj_stack, x0, bounds=bounds, approx_grad=True, messages=5, args=(x,y,z,i), epsilon=epsilon)
    #x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj_stack_gradient, x0, bounds=bounds, messages=5, args=(x,y,z,i), epsilon=epsilon)
    x_opt, _, _ = scipy.optimize.fmin_tnc(fitting_obj_stack_gradient, x0, bounds=bounds, messages=5, args=(x, y, z, i, loss), epsilon=epsilon)

    #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
    #x_opt = scipy.optimize.fmin(fitting_obj_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    
    ellipse_stack = {}
    cx, cy = x_opt[0], x_opt[1]
    wx, wy = x_opt[2], x_opt[3]

    print "cx, cy, wx, wy:", cx, cy, wx, wy

    radii = x_opt[4:]
    radii_x = radii[0:num_layers]
    radii_y = radii[num_layers:]
   
    assert len(radii_x) == len(radii_y) == num_layers

    # compile return datastructure
    for r_idx in xrange(num_layers):
        z_idx = r_idx + min_z
        tmp_cx = cx + r_idx*wx
        tmp_cy = cy + r_idx*wy
        ellipse_stack[z_idx] = Ellipse(tmp_cx, tmp_cy, z_idx, radii_x[r_idx], radii_y[r_idx], 0)

    return ellipse_stack


        
if __name__ == "__main__":

    print "checking gradient of abs loss"
    check_gradient()

    
    #fit = fit_ellipse_stack_scipy(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack(dx, dy, dz, di)
    #fit1 = fit_ellipse_stacked_squared(dx, dy, dz, di)

