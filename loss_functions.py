"""
number of loss functions to use with different models
"""

import numpy


def abs_loss(residual):
    """
    absolute loss
    """

    return numpy.abs(residual)



def squared_loss(residual):
    """
    squared loss
    """

    return residual * residual



def eps_loss(residual, epsilon):
    """
    epsilon-insensitive loss
    """

    if numpy.abs(residual) < epsilon:
        return 0
    else:
        return numpy.abs(residual) - epsilon


def eps_loss_bounded(residual, epsilon):
    """
    epsilon-insensitive loss
    """

    if numpy.abs(residual) < epsilon:
        return 0
    else:
        return max(numpy.abs(residual) - epsilon, 3)


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


