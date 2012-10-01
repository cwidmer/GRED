#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
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

        values = {}
        values["radius"] = self.radius.value()
        values["threshold"] = self.threshold.value()
        values["method_index"] = self.method.currentIndex()

        return values

