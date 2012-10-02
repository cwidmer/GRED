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
@summary: Interactive histogram using matplotlib

"""

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class HistogramQWidget(QtGui.QWidget):
    """
    QT wrapper for matplotlib
    """

    def __init__(self, parent=None):
        """
        setup GUI
        """


        self.data = None
        self.value = 0

        QtGui.QWidget.__init__(self, parent)
        self.my_layout = QtGui.QGridLayout(self)
        self.my_layout.setMargin(0)
        self.my_layout.setSpacing(10)


        # set up matplotlib
        self.dpi = 100
        self.fig = Figure((6.0, 6.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(300, 100)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        #self.canvas.setParent(self)
        self.my_layout.addWidget(self.canvas, 0, 0, 1, 2)
            
        self.axes = self.fig.add_subplot(111)
        self.cax = None

        # add spin box
        self.spin_label = QtGui.QLabel(self)#, 'Value:'
        self.my_layout.addWidget(self.spin_label, 1, 0)

        self.spin = QtGui.QDoubleSpinBox(self)
        self.spin.setMinimum(-10.0)
        self.spin.setMaximum(10.0)
        self.spin.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spin.setSingleStep(0.2)
        self.my_layout.addWidget(self.spin, 1, 1)


    def on_click(self, event):
        """
        process click from matplotlib
        """

        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
        self.emit(QtCore.SIGNAL('thresholdChanged(int)'), event.xdata)


    def update_value(self, v):
        """
        update plot

        """

        self.value = v
        self.update_plot()
 

    def update_dataset(self, dataset):
        """
        update plot

        """

        self.value = dataset.threshold
        self.data = dataset.red_channel.flatten()
        self.update_plot()
        

    def update_plot(self):
        """
        plotting routine
        """

        print "updating histogram plot"

        self.axes.clear()       
        self.axes.hist(self.data, bins=100, range=(min(self.data), max(self.data)))
        self.axes.axvline(x=self.value, linewidth=1, color='r')
        self.canvas.draw()

