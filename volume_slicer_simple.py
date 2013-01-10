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
@summary: Visualization of volume data and fits using matplotlib

"""

import copy
import pylab
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm

import util


class SimpleSlicerQWidget(QtGui.QWidget):
    """
    QT wrapper for matplotlib
    """

    def __init__(self, parent=None):
        """
        setup GUI
        """

        self.dataset = None
        self.ellipse_stack = None

        self.active_layer = 0

        QtGui.QWidget.__init__(self, parent)
        self.my_layout = QtGui.QVBoxLayout(self)
        self.my_layout.setMargin(0)
        self.my_layout.setSpacing(0)


        # set up matplotlib
        self.dpi = 100
        self.fig = Figure((6.0, 6.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(800, 400)
        #self.canvas.setParent(self)
        self.my_layout.addWidget(self.canvas) 
        
        self.axes = self.fig.add_subplot(121)
        self.axes_green = self.fig.add_subplot(122)
        self.cax = None

        #slider
        self.sld = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sld.setMinimum(0)
        #self.sld.setGeometry(30, 40, 100, 30)
        #sld.valueChanged[int].connect(self.changeValue)
        self.my_layout.addWidget(self.sld)

        self.connect(self.sld, QtCore.SIGNAL('valueChanged(int)'), self.update_active_layer)


    def update_dataset(self, dataset):
        """
        update plot

        """

        self.dataset = dataset
        new_max = self.dataset.volume.shape[2] - 1
        self.sld.setMaximum(new_max)

        self.update_plot()
        

    def update_active_layer(self, idx_z):
        """
        update active layer
        """

        self.active_layer = idx_z
        self.update_plot()


    def update_plot(self):
        """
        plotting routine
        """

        channel1 = self.dataset.volume[:,:,self.active_layer].T
        channel2 = self.dataset.green_channel[:,:,self.active_layer].T

        self.axes.clear()
        self.axes_green.clear()

        self.axes.imshow(channel1, interpolation="nearest")
        a = range(5)
        self.axes_green.imshow(channel2, interpolation="nearest", cmap=cm.Greys_r)


        if self.dataset.stack:
            n = 50
            z = int(self.active_layer)

            # check if ellipse is available for layer
            if self.dataset.stack.has_key(z):

                e = self.dataset.stack[z]
                dat = e.sample_equidistant(n)

                self.axes.plot(dat[0], dat[1], "b-", scalex=False, scaley=False)
                self.axes_green.plot(dat[0], dat[1], "b-", scalex=False, scaley=False)

                offset = self.dataset.radius_offset

                # plot manual offset in red
                if offset != 0:
                    e_off = copy.copy(e)
                    e_off.rx += offset
                    e_off.ry += offset

                    dat_off = e_off.sample_equidistant(n)
                    self.axes.plot(dat_off[0], dat_off[1], "r-", scalex=False, scaley=False)
                    self.axes_green.plot(dat_off[0], dat_off[1], "r-", scalex=False, scaley=False)


        self.canvas.draw()

