#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011-2013 Christian Widmer
# Copyright (C) 2011-2013 Max-Planck-Society, TU-Berlin, MSKCC

"""
@author: Christian Widmer
@summary: Some utility functions for the project, like sampling from an ellipse or ellipsoid

"""

import random
import numpy
import pylab


# def named tuple
#from collections import namedtuple
#Ellipse = namedtuple("Ellipse", ["cx", "cy", "cz", "rx", "ry", "alpha"])



class Ellipse(object):
    """
    ellipse class defining some plotting and sampling procedures
    """

    def __init__(self, cx, cy, cz, rx, ry, alpha):
        """
        constructor defining fields consisting of 
        center coordinates, radii and rotation angle
        """

        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.rx = rx
        self.ry = ry
        self.alpha = alpha


    def plot(self, num_points=100):
        """
        plot ellipse by sampling points
        """

        dat_x, dat_y = self.sample_equidistant(num_points)
        
        pylab.plot(dat_x, dat_y, "bo")
        pylab.show()


    def sample_equidistant(self, num_points):
        """
        sample from ellipsoid with equal angular spacing
        """

        theta = numpy.linspace(0, 2 * numpy.pi, num_points + 1)
    
        return self.sample_given_spacing(theta)


    def sample_uniform(self, num_points):
        """
        sample from ellipsoid with uniformly distributed spacing
        """

        theta = [random.uniform(0, 2 * numpy.pi) for _ in xrange(num_points + 1)]
    
        return self.sample_given_spacing(theta)


    def sample_given_spacing(self, theta):
        """
        sample points from ellipse given set of angles

        center: xc, yc
        radii: xr, yr
        rotation: alpha

        X = Z + Q(ALPHA) * [A * cos(theta); B * sin(theta)]
        where Q(ALPHA) is the rotation matrix
        Q(ALPHA) = [cos(ALPHA), -sin(ALPHA); 
                    sin(ALPHA), cos(ALPHA)]
        """

        # set up points
        points = numpy.zeros((2, len(theta)))
        points[0, :] = self.rx * numpy.cos(theta)
        points[1, :] = self.ry * numpy.sin(theta)

        ## get initial rotation matrix
        rot = numpy.array( [[ numpy.cos(self.alpha), -numpy.sin(self.alpha) ], 
                            [ numpy.sin(self.alpha),  numpy.cos(self.alpha) ]])
        
        ## center
        center = numpy.zeros((2, 1))
        center[0] = self.cx
        center[1] = self.cy

        # perform rotation
        dat = numpy.dot(rot , points) + center

        return dat


def ellipsoid(xc, yc, zc, xr, yr, zr, n=200):
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
    
    e = Ellipse(1,1,0,2,3,0)
    e.plot()

