#!/usr/bin/env python2.5
#
# Written (W) 2011 Christian Widmer
# Copyright (C) 2011 Max-Planck-Society

"""
@author: Christian Widmer

@summary: Code for fitting 2D circles

"""

import pylab
import scipy.optimize
import numpy
import os
import Image

data_x = [0.0, -1.0, 1.0,  0.0, 3.0]
data_y = [1.0,  0.0, 0.0, -1.0, 3.0]
data_intensity = numpy.ones(len(data_x))


#def epsilon_insensitive_loss(param, data, epsilon):
def fitting_obj(param):
    """
    computes residuals based on distance from circle
    
    can be used with different loss-functions on residual
    """

    obj = 0

    for idx in range(len(data_x)):
        residual = numpy.sqrt((param[0] - data_x[idx])**2 + (param[1] - data_y[idx])**2) - param[2]
        #obj += squared_loss(residual)
        #obj += eps_loss(residual, 2)*data_intensity[idx]
        #obj += eps_loss(residual, 2)
        obj += eps_loss_asym(residual, 2, 1.0, 0.3)

    return obj


def squared_loss(residual):
    
    return residual * residual



def eps_loss(residual, epsilon):
    """
    epsilon-insensitive loss
    """

    if numpy.abs(residual) < epsilon:
        return 0
    else:
        return numpy.abs(residual) - epsilon



def eps_loss_asym(residual, epsilon, slope_inside, slope_outside):
    """
    asymmetric loss for inside and outside of the circle
    """

    if numpy.abs(residual) < epsilon:
        return 0
    elif residual > 0:
        return (residual - epsilon) * slope_outside
    else:
        return (numpy.abs(residual) - epsilon) * slope_inside



def load_tif(threshold, hist):
    """
    load data file
    """

    tif_dir = "data/20091026_SK570_578_4.5um_1_R3D_CAL_02_D3D_CPY_Mad1/"
    tiffs = [tif_dir + f for f in os.listdir(tif_dir) if f.endswith(".tif")]
    tiffs.sort()

    # convert to numpy array
    myimg = numpy.asarray(Image.open(tiffs[10]).convert("RGB"))

    # extract red channel
    red_channel = myimg[:,:,0]

    # perform thresholding
    b = red_channel.flatten()

    if hist:
        pylab.hist(b, bins=30)
        pylab.show()

    b[b < threshold] = 0
    b = b.reshape(red_channel.shape)

    

    # show image
    pylab.figure()
    pylab.imshow(b.T)
    #pylab.show()

    # set up alternate representation
    data_x = []
    data_y = []
    data_intensity = []

    for idx_x in range(b.shape[0]):
        for idx_y in range(b.shape[1]):
            if b[idx_x, idx_y] > 0:
                assert(b[idx_x, idx_y] == red_channel[idx_x, idx_y])
                data_x.append(idx_x)
                data_y.append(idx_y)
                data_intensity.append(b[idx_x, idx_y])

    return data_x, data_y, data_intensity



if __name__ == "__main__":

    x0 = numpy.array([10, 10, 10])

    #x_opt = scipy.optimize.fmin(fitting_obj, x0)
    #print x_opt

    #pylab.plot(data_x, data_y, "o")
    #cir = pylab.Circle((x_opt[0],x_opt[1]), radius=x_opt[2], alpha=0.2)
    #pylab.gca().add_patch(cir)
    #pylab.axis('scaled')
    #pylab.show()


    data_x, data_y, data_intensity = load_tif(160, True)

    x_opt = scipy.optimize.fmin(fitting_obj, x0)
    print x_opt
    pylab.plot(data_x, data_y, "o")

    cir = pylab.Circle((x_opt[0], x_opt[1]), radius=x_opt[2], alpha=1.0)
    pylab.gca().add_patch(cir)
    pylab.axis('scaled')
    pylab.show()

