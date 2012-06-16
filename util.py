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
@summary: Some utility functions for the project, like sampling from an ellipse or ellipsoid

"""


import numpy
import pylab
from collections import namedtuple


# def named tuple
Ellipse = namedtuple("Ellipse", ["cx", "cy", "cz", "rx", "ry", "alpha"])


class EllipseXX(object):
    """
    simple container for ellipse data
    """

    def __init__(self, cx, cy, cz, rx, ry, alpha):
        """
        constructor
        """

        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.rx = rx
        self.ry = ry
        self.alpha = alpha



def ellipse(xc, yc, xr, yr, alpha, n=200):
    """
    sample points from ellipse

    center: xc, yc
    radii: xr, yr
    rotation: alpha
    %       X = Z + Q(ALPHA) * [A * cos(theta); B * sin(theta)]
    %       where Q(ALPHA) is the rotation matrix
    %       Q(ALPHA) = [cos(ALPHA), -sin(ALPHA); 
    %                   sin(ALPHA), cos(ALPHA)]
    """

    pi = numpy.pi
    theta = numpy.linspace (0, 2 * pi, n + 1);

    points = numpy.zeros((2, n+1))
    points[0,:] = xr * numpy.cos(theta);
    points[1,:] = yr * numpy.sin(theta);

    ## Get initial rotation matrix
    rm = numpy.array( [[ numpy.cos(alpha), -numpy.sin(alpha) ], [ numpy.sin(alpha), numpy.cos(alpha) ]] )
    
    ## center
    center = numpy.zeros((2, 1))
    center[0] = xc
    center[1] = yc

    # perform rotation
    dat = numpy.dot(rm , points) + center

    return dat


def ellipsoid(xc, yc, zc, xr, yr, zr, n=200, plot=False):
    """
    sample points from ellipsoid

    note: does not support rotation
    """

    pi = numpy.pi
    theta = numpy.linspace (0, 2 * pi, n + 1);
    phi = numpy.linspace (-pi / 2, pi / 2, n + 1);
    [theta, phi] = numpy.meshgrid (theta, phi);

    lx = xr * numpy.cos(phi) * numpy.cos(theta) + xc;
    ly = yr * numpy.cos(phi) * numpy.sin(theta) + yc;
    lz = zr * numpy.sin(phi) + zc;

    return lx.flatten(), ly.flatten(), lz.flatten()


def plot_point_cloud(x,y,z):
    """
    plot 3d point cloud
    """

    import mpl_toolkits.mplot3d.axes3d as p3
    fig = pylab.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(x,y,z)

    pylab.show()


def plot_ellipse(cx, cy, cz, rx, ry, alpha, figure):
    """
    plot ellipse stack
    """


    from mayavi import mlab

    n = 50


    dat = ellipse(cx, cy, rx, ry, alpha, n)
    #dat = ellipse(e.cx, e.cy, e.rx, e.ry, e.alpha, n)
    
    dx = dat[0]
    dy = dat[1]
    dz = [cz]*(n+1)
    dv = [25]*(n+1)

    return mlab.plot3d(dx,dy,dz,dv,
                        #mode='2dcircle',
                        #mode='point',
                        color=(0, 0, 0),
                        #scale_factor=100*max(self.data.shape),
                        figure=figure,
                        line_width=20,
                        tube_radius=None
                    )


def plot_ellipse_stack(ellipse_stack, figure):
    """
    plot ellipse stack
    """

    for e in ellipse_stack:
        plot_ellipse(e, figure)



if __name__ == "__main__":
    
    e = ellipse(3,6,2,3,0)

    print numpy.average(e[0])
    print numpy.average(e[1])


