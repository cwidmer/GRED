#!/usr/bin/env python2.5
#
# Written (W) 2012 Christian Widmer
# Copyright (C) 2012 Max-Planck-Society

"""
@author: Christian Widmer

@summary: dialog for preprocessing

"""

from PyQt4 import QtCore, QtGui, uic
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, 
                                   QtGui.QSizePolicy.Expanding, 
                                   QtGui.QSizePolicy.Expanding)

        FigureCanvas.updateGeometry(self)


class MplWidget(QtGui.QWidget):
    """
    thin Qt wrapper of matplotlib
    """

    def __init__(self, parent = None):

        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)



class PreprocDialog(QtGui.QDialog):
    """
    simple dialog to fire up batch processing
    """

    def __init__(self,parent=None):
        """
        setup gui from ui file (edit with qt designer)
        """

        QtGui.QDialog.__init__(self)
        self.ui = uic.loadUi('preproc.ui', self)
        self.ui.show()


    def getValues(self):
        """
        wraps up return values as dictionary
        """

        # naming has to agree with method "fit_stack" in qt_gui.py
        method_idx_to_name = ["squared", "eps", "circle"]

        values = {}
        values["radius_offset"] = self.radius_offset.value()
        values["percentile"] = self.percentile.value()
        values["std_cut"] = self.std_cut.value()
        values["method"] = method_idx_to_name[self.method.currentIndex()]

        return values


