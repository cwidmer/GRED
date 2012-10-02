#!/usr/bin/env python2.5
#
# Written (W) 2012 Christian Widmer
# Copyright (C) 2012 Max-Planck-Society

"""
@author: Christian Widmer

@summary: dialog for batch processing

"""

from PyQt4 import QtCore, QtGui, uic


class BatchDialog(QtGui.QDialog):
    """
    simple dialog to fire up batch processing
    """

    def __init__(self,parent=None):
        """
        setup gui from ui file (edit with qt designer)
        """

        QtGui.QDialog.__init__(self)
        self.ui = uic.loadUi('batch_dialog.ui', self)
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

