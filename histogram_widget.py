#!/usr/bin/env python2.5
#
# Written (W) 2011-2012 Christian Widmer
# Copyright (C) 2011-2012 Max-Planck-Society

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
        self.spin_label = QtGui.QLabel(self)
        self.spin_label.setText('Value:')
        self.my_layout.addWidget(self.spin_label, 1, 0)

        self.spin = QtGui.QDoubleSpinBox(self)
        self.spin.setMinimum(0.0)
        self.spin.setMaximum(1000.0)
        #self.spin.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spin.setSingleStep(10)
        self.spin.setKeyboardTracking(False)
        self.spin.setReadOnly(False)
        self.my_layout.addWidget(self.spin, 1, 1)

        # connect signals
        self.connect(self.spin, QtCore.SIGNAL('valueChanged(double)'), self.update_value)


    def on_click(self, event):
        """
        process click from matplotlib
        """

        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
        self.update_value(float(event.xdata))


    def update_value(self, v):
        """
        update plot

        """

        self.value = v
        self.spin.setValue(self.value)
        self.update_plot()

        self.emit(QtCore.SIGNAL('thresholdChanged(double)'), self.value)
 

    def update_dataset(self, dataset):
        """
        wrapper for update value
        
        """
        #TODO this can go in subclass
        self.value = dataset.threshold
        self.data = dataset.red_channel.flatten()
        self.update_plot()
        self.spin.setValue(self.value)
        

    def update_plot(self):
        """
        plotting routine
        """

        print "updating histogram plot"

        self.axes.clear()       
        self.axes.hist(self.data, bins=100, range=(min(self.data), max(self.data)))
        self.axes.axvline(x=self.value, linewidth=1, color='r')
        self.canvas.draw()

