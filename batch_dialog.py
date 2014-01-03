#!/usr/bin/env python2.5
#
# Written (W) 2012-2014 Christian Widmer
# Copyright (C) 2012-2014 Max-Planck-Society, MSKCC, TU-Berlin


"""
@author: Christian Widmer

@summary: dialog for batch processing

"""

from PySide import QtCore, QtGui, QtUiTools


class BatchDialog(QtGui.QDialog):
    """
    simple dialog to fire up batch processing
    """

    def __init__(self,parent=None):
        """
        setup gui from ui file (edit with qt designer)
        """

        QtGui.QDialog.__init__(self)

        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile('batch_dialog.ui')
        uifile.open(QtCore.QFile.ReadOnly)

        self.ui = loader.load(uifile, self)
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

