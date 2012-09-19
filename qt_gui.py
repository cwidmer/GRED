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
@summary: Visualization of the fitted ellipsoid using PyQt4 and mayavi2

"""

# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
import cPickle
os.environ['ETS_TOOLKIT'] = 'qt4'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
#from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#import sip
#sip.setapi('QString', 1)
from PyQt4 import QtGui, QtCore
import numpy
import util
import scipy.stats
from collections import namedtuple

from traits.api import HasTraits, Instance, on_trait_change, Tuple, Dict #List, Int, Array
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

from fit_ellipsoid import fit_ellipsoid
import fit_ellipse_stack
import fit_ellipse_stack_conic
#import fit_cone_stack_cvxpy
from fit_sphere import fit_sphere_stack

from data_processing import artificial_data, load_tif, threshold_volume #, generate_sphere_full
from volume_slicer_simple import SimpleSlicerQWidget
from histogram_widget import HistogramQWidget



################################################################################
#The actual visualization
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    data = Tuple()
    #data = Array()
    ellipse_stack = Dict() #List()

    data_plot = None
    stack_plot = None


    @on_trait_change('scene.activated')
    def update_plot(self):
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.
        print "init stuff"

        # init data plot
        
        self.data_plot = self.scene.mlab.test_contour3d()
        #self.scene.mlab.points3d(x,y,z,v,colormap="copper", scale_factor=.0025, opacity=0.4)
        #v = numpy.ones((3,3,3))
        #self.data_plot = self.scene.mlab.contour3d(v, opacity=0.2)

        # init ellipse stack plot
        #dx = dy = dz = dv = []
        #self.stack_plot = self.scene.mlab.points3d(dx,dy,dz,dv,colormap="copper", scale_factor=.01, opacity=0.6)


    @on_trait_change('ellipse_stack')
    def update_ellipse_stack(self):
        """
        plot ellipse stack
        """


        self.update_data()
        n = 50
        
        # sample data
        for e in self.ellipse_stack.values():
             
            dat = util.ellipse(e.cx, e.cy, e.rx, e.ry, e.alpha, n)
            
            dx = dat[0]
            dy = dat[1]
            dz = [e.cz]*(n+1)
            dv = [25]*(n+1)

            self.scene.mlab.points3d(dx,dy,dz,dv,colormap="copper", scale_factor=.015, opacity=0.5)

        # update data source
        # TODO keep list of plots
        #self.stack_plot.mlab_source.set(x=dx, y=dy, z=dz, s=dv)


    @on_trait_change('data')
    def update_data(self):
        """
        update ellipse stack

        list of available colormaps (http://github.enthought.com/mayavi/mayavi/mlab.html):

        accent       flag          hot      pubu     set2
        autumn       gist_earth    hsv      pubugn   set3
        black-white  gist_gray     jet      puor     spectral
        blue-red     gist_heat     oranges  purd     spring
        blues        gist_ncar     orrd     purples  summer
        bone         gist_rainbow  paired   rdbu     winter
        brbg         gist_stern    pastel1  rdgy     ylgnbu
        bugn         gist_yarg     pastel2  rdpu     ylgn
        bupu         gnbu          pink     rdylbu   ylorbr
        cool         gray          piyg     rdylgn   ylorrd
        copper       greens        prgn     reds
        dark2        greys         prism    set1
        """

        print "updating data"
        self.scene.mlab.clf()

        # update data source
        #self.data_plot = self.scene.mlab.contour3d(self.data[0], self.data[1], self.data[2], self.data[3], opacity=0.2)
        #self.data_plot = self.scene.mlab.contour3d(self.data[0], self.data[1], self.data[2], self.data[3], opacity=0.2)
        #self.data_plot = self.scene.mlab.contour3d(self.data, opacity=0.2)

        # look at: http://github.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
        #from mayavi import mlab
        #self.data_plot = mlab.pipeline.volume(mlab.pipeline.scalar_field(self.volume))

        self.data_plot = self.scene.mlab.points3d(self.data[0], self.data[1], self.data[2], self.data[3],colormap="Reds", scale_factor=.001, opacity=0.2, scale_mode="scalar")

        #self.data_plot.mlab_source.set(scalars=self.data, opacity=0.1)
        


    #TODO fix
    #@on_trait_change('ellipsoid')
    def plot_ellipsoid(self, cx, cy, cz, rx, ry, rz):
        """
        plot ellispoid given three center coordinates (cx, cy, cz) 
        and three radii (rx, ry, rz)
        """

        n = 15

        # debug
        #######################################
        x = [cx]
        y = [cy]
        z = [cz]
        v = [100]
        print "x,y,z=%f,%f,%f" % (cx, cy, cz) 
        self.scene.mlab.points3d(x,y,z,v,colormap="copper", scale_factor=.025, opacity=0.4)
        #######################################

        pi = numpy.pi
        theta = numpy.linspace (0, 2 * pi, n + 1);
        phi = numpy.linspace (-pi / 2, pi / 2, n + 1);
        [theta, phi] = numpy.meshgrid (theta, phi);

        lx = rx * numpy.cos(phi) * numpy.cos(theta) + cx;
        ly = ry * numpy.cos(phi) * numpy.sin(theta) + cy;
        lz = rz * numpy.sin(phi) + cz;
        lv = numpy.ones(lz.shape)
        #self.scene.mlab.plot3d(lx.flatten(), ly.flatten(), lz.flatten(), lv.flatten(), opacity=0.3)
        #self.scene.mlab.contour3d(lx, ly, lz, lv, opacity=0.3)
        self.scene.mlab.mesh(lx, ly, lz, scalars=lv, opacity=0.2, representation="wireframe", line_width=1.0)


    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=450, width=500, show_label=False),
                resizable=True # We need this to resize with the parent widget
                )


################################################################################
# The QWidget containing the visualization, this is pure PyQt4 code.
class VolumeSlicerQWidget(QtGui.QWidget):
    """
    QT wrapper for mayavi2
    """

    def __init__(self, parent=None):
        """
        setup GUI
        """

        QtGui.QWidget.__init__(self, parent)
        self.my_layout = QtGui.QVBoxLayout(self)
        self.my_layout.setMargin(0)
        self.my_layout.setSpacing(0)

        x, y, z, i, volume = artificial_data()
        self.slicer = VolumeSlicer(data=volume)

        # The edit_traits call will generate the widget to embed.
        self.ui = self.slicer.edit_traits(parent=self, kind='subpanel').control
        self.my_layout.addWidget(self.ui)
        self.ui.setParent(self)


    def update_dataset(self, dataset):
        """
        update plot

        """
        #TODO it would be preferable to just update the data

        self.slicer.data = dataset.volume

        return ""

        self.ui.setParent(None)

        print "VolumeSlicerQWidget: update dataset"
        self.slicer = VolumeSlicer(data=dataset.volume)

        # The edit_traits call will generate the widget to embed.
        self.ui = self.slicer.edit_traits(parent=self, kind='subpanel').control
        #layout.addWidget(self.ui)
        self.ui.setParent(self)

        self.my_layout.addWidget(self.ui)

        self.ui.setParent(self)
    
        #if dataset.volume != None:
        #    self.slicer.data = dataset.volume

        #if dataset.stack != None:
        #    self.slicer.ellipse_stack = dataset.stack


    def update_ellipse_stack(self, dataset):
        """
        dedicated method to update stack fit
        """

        if dataset.stack != None:
            self.slicer.ellipse_stack = dataset.stack


class MayaviQWidget(QtGui.QWidget):
    """
    QT wrapper for 3d plot
    """

    def __init__(self, parent=None):
        """
        setup GUI
        """

        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


    def update_dataset(self, dataset):
        """
        update plot
        """
   
        print "MayaviQWidget: update dataset"

        if dataset.volume != None:
            self.visualization.data = tuple(dataset.points)
            #self.visualization.data = dataset.volume
        if dataset.stack != None:
            self.visualization.ellipse_stack = dataset.stack



class ControlWidget(QtGui.QWidget):
    """
    widget to hold control buttons
    """
  
    def __init__(self):
        super(ControlWidget, self).__init__()

        self.initUI()
        self.directory = None
        self.super_directory = None

    def initUI(self):
        """
        set up gui elements and layout
        """

        # set up layout
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignLeading)
        self.layout.setMargin(0)
        self.layout.setSpacing(0)

        # add buttons
        self.button_dat = QtGui.QPushButton('Add Dataset', self)
        self.button_dat.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_dat)   

        self.button_all_dat = QtGui.QPushButton('Add all Datasets', self)
        self.button_all_dat.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_all_dat)

        self.button_load = QtGui.QPushButton('Load', self)
        self.button_load.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_load)

        self.button_save = QtGui.QPushButton('Save', self)
        self.button_save.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_save)

        #self.button_fit = QtGui.QPushButton('Fit ellipsoid', self)
        #self.button_fit.setFocusPolicy(QtCore.Qt.NoFocus)
        #self.layout.addWidget(self.button_fit)

        self.button_fit_stack = QtGui.QPushButton('Fit ellipse stack (squared)', self)
        self.button_fit_stack.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_fit_stack)

        #TODO re-enable if working
        self.button_fit_insensitive = QtGui.QPushButton('Fit ellipse stack (abs)', self)
        self.button_fit_insensitive.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_fit_insensitive)

        self.button_fit_sphere_stack = QtGui.QPushButton('Fit circle stack (abs)', self)
        self.button_fit_sphere_stack.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_fit_sphere_stack)

        #self.button_batch = QtGui.QPushButton('Batch process', self)
        #self.button_batch.setFocusPolicy(QtCore.Qt.NoFocus)
        #self.layout.addWidget(self.button_batch)

        self.button_eval = QtGui.QPushButton('Evaluate', self)
        self.button_eval.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_eval)

        self.button_export = QtGui.QPushButton('Export', self)
        self.button_export.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.button_export)

        #'Adapt Radius', 
        self.spin_radius_label = QtGui.QLabel("Radius Offset", self)
        self.layout.addWidget(self.spin_radius_label)

        self.spin_radius = QtGui.QDoubleSpinBox(self)
        self.spin_radius.setMinimum(-10.0)
        self.spin_radius.setMaximum(10.0)
        self.spin_radius.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spin_radius.setSingleStep(0.2)
        self.layout.addWidget(self.spin_radius)

        # connect signals
        self.connect(self.button_dat, QtCore.SIGNAL('clicked()'), self.showDialog)
        self.connect(self.button_all_dat, QtCore.SIGNAL('clicked()'), self.select_super_dir)
        self.setFocus()
        
        self.setWindowTitle('Select Files')
        #self.setGeometry(300, 300, 350, 80)
        
    
    def showDialog(self):
        self.directory = QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        self.emit(QtCore.SIGNAL('directoryChanged(PyQt_PyObject)'), self.directory)

    def select_super_dir(self):
        self.super_directory = QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        self.emit(QtCore.SIGNAL('superDirectoryChanged(PyQt_PyObject)'), self.super_directory)


class TableWidget(QtGui.QTableWidget):
    """
    widget class to hold table information (including the data directories)
    """

    def __init__(self):
        super(TableWidget, self).__init__()

        self.datasets = []
        self.initUI()

        # provide clean interface to outside world
        self.connect(self, QtCore.SIGNAL('itemClicked(QTableWidgetItem*)'), self.emit_dataset)


    def initUI(self):
        """
        set up GUI
        """

        self.setColumnCount(7)
        self.setHorizontalHeaderLabels(["file name", "radius", "threshold", "num pixels", "area (micro m^2)", "total intensity", "intensity per area"]);
        self.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch);

        self.setSelectionBehavior(1) #select only rows
        self.setShowGrid(1)



    def add_dataset(self, dat):
        """
        appends dat to dataset and adds new table item
        """

        fn = dat.split(os.sep)[-1]

        item = QtGui.QTableWidgetItem(fn)
        item.dataset = dat
        row = self.rowCount()

        self.insertRow(row)
        self.setItem(row, 0, item)
        self.setItem(row, 1, QtGui.QTableWidgetItem("-"))
        self.setItem(row, 2, QtGui.QTableWidgetItem("-"))
        self.setItem(row, 3, QtGui.QTableWidgetItem("-"))
        self.setItem(row, 4, QtGui.QTableWidgetItem("-"))
        self.setItem(row, 5, QtGui.QTableWidgetItem("-"))
        self.setItem(row, 6, QtGui.QTableWidgetItem("-"))
        self.datasets.append(dat)
        self.selectRow(row)


    def emit_dataset(self, item):
        """
        re-emit only the directory name
        """

        print "item", item

        self.emit(QtCore.SIGNAL('directoryChanged(PyQt_PyObject)'), self.datasets[item.row()])


    def update_evaluation(self, dataset):
        """
        set new eval data
        """

        evaluation = dataset.evaluation

        if evaluation:

            row = self.currentRow()

            print "current row", row

            self.item(row, 1).setText("%.2f" % dataset.radius_offset)
            self.item(row, 2).setText("%.2f" % dataset.threshold)
            self.item(row, 3).setText("%d"   % evaluation.total_num_pixels)
            self.item(row, 4).setText("%.2f" % evaluation.total_area_in_micro_m)
            self.item(row, 5).setText("%.2f" % evaluation.total_intensity)
            self.item(row, 6).setText("%.2f" % evaluation.total_intensity_per_area)


class MainWidget(QtGui.QTreeWidget):
    """
    main widget
    """


    def __init__(self):
        """
        setup up main gui layout
        """

        #####
        # non-gui variables
        #####
        self.datasets = {}
        self.active_dataset = None
        #

        super(MainWidget, self).__init__()
        self.setWindowTitle("Cell tracker")
        # define a "complex" layout to test the behaviour
        layout = QtGui.QGridLayout(self)

        # set up gui
        label_slicer = QtGui.QLabel(self)
        label_slicer.setText("Volume Slicer")
        label_slicer.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        layout.addWidget(label_slicer, 0, 0)

        slicer_widget = SimpleSlicerQWidget(self)
        layout.addWidget(slicer_widget, 1, 0)

        label_view = QtGui.QLabel(self)
        label_view.setText("Volume View")
        label_view.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        layout.addWidget(label_view, 0, 1)

        mayavi_widget = MayaviQWidget(self)
        layout.addWidget(mayavi_widget, 1, 1)

        table_widget = TableWidget()
        layout.addWidget(table_widget, 2, 0)

        hist_widget = HistogramQWidget()
        layout.addWidget(hist_widget, 2, 1)

        control_widget = ControlWidget()
        layout.addWidget(control_widget, 2, 2)
        #control_widget.show()


        self.connect(control_widget, QtCore.SIGNAL('directoryChanged(PyQt_PyObject)'), self.add_dataset)

        self.connect(self, QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), mayavi_widget.update_dataset)
        self.connect(self, QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), slicer_widget.update_dataset)
        self.connect(self, QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), hist_widget.update_dataset)
        self.connect(self, QtCore.SIGNAL('newKey(PyQt_PyObject)'), table_widget.add_dataset)
        self.connect(table_widget, QtCore.SIGNAL('directoryChanged(PyQt_PyObject)'), self.change_active_dataset)
        self.connect(hist_widget, QtCore.SIGNAL('thresholdChanged(int)'), self.update_threshold)
        self.connect(self, QtCore.SIGNAL('activeDatasetEvaluated(PyQt_PyObject)'), table_widget.update_evaluation)

        # create deep links to control widget (simple)
        #self.connect(control_widget.button_fit, QtCore.SIGNAL('clicked()'), mayavi_widget.update_ellipsoid)
        self.connect(control_widget.button_fit_sphere_stack, QtCore.SIGNAL('clicked()'), self.update_stack)
        self.connect(control_widget.button_fit_stack, QtCore.SIGNAL('clicked()'), self.update_ellipse_stack)
        #TODO fix an re-enable
        self.connect(control_widget.button_fit_insensitive, QtCore.SIGNAL('clicked()'), self.update_ellipse_stack_eps)
        self.connect(control_widget.button_load, QtCore.SIGNAL('clicked()'), self.load)
        self.connect(control_widget.button_eval, QtCore.SIGNAL('clicked()'), self.evaluate)
        self.connect(control_widget.button_export, QtCore.SIGNAL('clicked()'), self.export)
        self.connect(control_widget.button_save, QtCore.SIGNAL('clicked()'), self.save)
        self.connect(control_widget.spin_radius, QtCore.SIGNAL('valueChanged(double)'), self.update_radius_offset)
        self.connect(control_widget, QtCore.SIGNAL('superDirectoryChanged(PyQt_PyObject)'), self.add_all_datasets)

        # TODO remove this eventually
        #super_dir = "/fml/ag-raetsch/home/sumrania/Desktop/forChris_Philipp/"

        #self.add_all_datasets(super_dir)

        self.show()
 

    def add_all_datasets(self, super_dir):
        """
        appends all directories in super_dir
        """

        super_dir = str(super_dir)

        directories = [os.path.join(super_dir,dat) for dat in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir,dat))]
        directories.sort()

        for dat in directories:
            self.add_dataset(dat)      


    def add_dataset(self, tif_dir):
        """
        slot add_dataset
        """
        
        dataset = Dataset(tif_dir)
        dataset.load_data()
        self.datasets[tif_dir] = dataset

        self.emit(QtCore.SIGNAL('newKey(PyQt_PyObject)'), tif_dir)
        
        # notify observers
        self.change_active_dataset(tif_dir)


    def change_active_dataset(self, key):
        """
        emits signal that indicates that the currently active dataset was changed
        """

        self.active_dataset = self.datasets[key]

        id = QtCore.QMetaType.type('Dataset')
        self.emit(QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), self.active_dataset)

    
    def update_stack(self):
        """
        invoke fit, call update
        """

        print "updating stack"

        self.active_dataset.fit_stack("circle")
        self.emit(QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), self.active_dataset)


    def update_ellipse_stack(self):
        """
        invoke fit, call update
        """

        print "updating stack"

        self.active_dataset.fit_stack("squared")
        self.emit(QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), self.active_dataset)


    def update_ellipse_stack_eps(self):
        """
        invoke fit, call update
        """

        print "updating stack"

        self.active_dataset.fit_stack("eps")
        self.emit(QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), self.active_dataset)


    def update_threshold(self, thres):
        """
        emits signal that indicates that the currently active dataset was changed
        """

        print "updating threshold"
        self.active_dataset.update_threshold(thres)
        self.emit(QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), self.active_dataset)


    def update_radius_offset(self, offset):
        """
        emits signal that indicates that the currently active dataset was changed
        """

        print "updating threshold"
        self.active_dataset.update_radius_offset(offset)
        self.emit(QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), self.active_dataset)


    def evaluate(self):
        """
        emits signal that indicates that the currently active dataset was changed
        """
        
        self.active_dataset.evaluate()
        self.emit(QtCore.SIGNAL('activeDatasetEvaluated(PyQt_PyObject)'), self.active_dataset)


    def export(self):
        """
        emits signal that indicates that the currently active dataset was changed
        """
        
        dialog = QtGui.QFileDialog()
        #dialog.setFileMode(QtGui.QFileDialog.ShowDirsOnly)
        dir_name = str(dialog.getExistingDirectory(self, 'Select Directory'))

        file_name = dir_name + os.sep + "export.csv" 

        f = file(file_name, "w")
        f.write("file name, radius, threshold, num pixels, area (micro m^2), total intensity, intensity per area\n")

        for dataset_path, dataset in self.datasets.items():

            #dat_name = dataset_path.split(os.sep)[-1]
            dat_name = dataset_path.split("/")[-1]

            line = dat_name + ", "
            line += str(dataset.radius_offset) + ", "
            line += str(dataset.threshold) + ", "

            if dataset.evaluation != None:
                line += str(dataset.evaluation.total_num_pixels) + ", "
                line += str(dataset.evaluation.total_area_in_micro_m) + ", "
                line += str(dataset.evaluation.total_intensity) + ", "
                line += str(dataset.evaluation.total_intensity_per_area)

                # write separate file
                #inner_file_name = dir_name + os.sep + dat_name + ".csv"
                inner_file_name = dir_name + "/" + dat_name + ".csv"
                inner_f = file(inner_file_name, "w")
                inner_f.write("layer_id, area_micro_m, intensity\n")
                print "writing file", inner_file_name
 
                for layer in xrange(dataset.evaluation.num_layers):
                    area = dataset.evaluation.layer_area_in_micro_m[layer]
                    intensity = dataset.evaluation.layer_intensity[layer]
                    inner_f.write("%i, %f, %f\n" % (layer, area, intensity))

                inner_f.close()

            line += "\n"
            f.write(line)

        f.close()

        print "file successfully written to", file_name


    def save(self):
        """
        emits signal that indicates that the currently active dataset was changed
        """
        
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.AnyFile)
        file_name = str(dialog.getSaveFileName(self, 'Save project', "cell_fit.proj"))

        if not file_name == "":
            try:
                f = file(file_name, "w")
                self.datasets = cPickle.dump(self.datasets, f)
                f.close()

            except Exception, detail:
                print "error on saving file", file_name
                print detail

            print "successfully saved project to", file_name


    def load(self):
        """
        emits signal that indicates that the currently active dataset was changed
        """

        dialog = QtGui.QFileDialog()
        #dialog.setFileMode(QtGui.QFileDialog.AnyFile)
        file_name = str(dialog.getOpenFileName(self, 'Load project'))

        if not file_name == "":
            try:
                f = file(file_name)
                self.datasets = cPickle.load(f)

                for tif_dir, dataset in self.datasets.items():
                     self.emit(QtCore.SIGNAL('newKey(PyQt_PyObject)'), tif_dir)
                     self.active_dataset = dataset
                     self.emit(QtCore.SIGNAL('activeDatasetChanged(PyQt_PyObject)'), self.active_dataset)
                     self.emit(QtCore.SIGNAL('activeDatasetEvaluated(PyQt_PyObject)'), self.active_dataset)
                f.close()

            except Exception, detail:
                print "error on loading file", file_name
                print detail

            print "successfully loaded project from", file_name


# define class Data
Data = namedtuple("Data", ["x", "y", "z", "i"])


class EvaluationData(object):
    """
    class to hold information about evaluation
    """


    def __init__(self):

        # evaluation
        self.total_num_pixels = 0
        self.total_area_in_micro_m = 0.0
        self.total_intensity = 0.0
        self.total_intensity_per_area = 0.0

        self.layer_num_pixels = []
        self.layer_area_in_micro_m = []
        self.layer_intensity = []
        self.layer_intensity_per_area = []

        self.num_layers = 0


    def add_layer(self, num_pixels, intensity):
       
        area = float(num_pixels) * 0.04626801

        self.layer_num_pixels.append(num_pixels)
        self.layer_area_in_micro_m.append(area)
        self.layer_intensity.append(intensity)
        self.layer_intensity_per_area.append(float(intensity)/float(area))

        # update totals
        self.total_num_pixels += num_pixels
        self.total_area_in_micro_m += area
        self.total_intensity += intensity
        self.total_intensity_per_area = float(self.total_intensity) / float(self.total_area_in_micro_m)

        self.num_layers += 1


class Dataset(object):
    """
    class to hold information about one dataset
    """


    def __init__(self, tif_dir):
        """
        set up dataset
        """

        self.tif_dir = tif_dir
        self.threshold = 255

        # set defaults
        self.clear()


    def clear(self):
        """
        set defaults
        """

        self.red_channel = None
        self.green_channel = None
        self.points = None
        self.volume = None
        self.is_loaded = False
        self.stack = None
        self.ellipsoid = None
        self.radius_offset = 0        

        self.evaluation = None


    def load_data(self):
        """
        load raw data 
        """

        self.red_channel = load_tif(self.tif_dir, "w617")
        self.green_channel = load_tif(self.tif_dir, "w528")
        #x, y, z, i, vol = generate_sphere_full()
        #print "WARNING: DEBUG"
        #self.green_channel = vol

        # set threshold to some high percentile by default
        self.threshold = scipy.stats.scoreatpercentile(self.red_channel.flatten(), 93)

        self.volume_to_points()

        return self


    def volume_to_points(self):
        """
        convert volume to points
        """

        x, y, z, i, vol = threshold_volume(self.red_channel, self.threshold) 
        #print "WARNING DEBUGGIN"
        #x, y, z, i, vol = artificial_data() #artificial_data()
        self.points = Data(x, y, z, i)
        self.volume = vol
        self.is_loaded = True

        print "new thresholding done"


    def update_threshold(self, thres):
        """
        update threshold
        """

        self.threshold = thres
        self.volume_to_points()


    def update_radius_offset(self, offset):
        """
        update threshold
        """

        print "update_radius_offset", offset, type(offset)
        self.radius_offset = offset


    ###################################
    # glue data to fitting code

    def fit_ellipsoid(self):
        """
        invokes code to fit ellipsoid
        """

        if self.is_loaded:
            self.ellipsoid = fit_ellipsoid(self.points.x, self.points.y, self.points.z, self.points.v, 500)

        return self


    def fit_stack(self, method):
        """
        invokes code to fit stack of ellipses
        """

        if self.is_loaded:
            print "fitting stack using method", method


            if method == "circle":
                self.stack = fit_sphere_stack(self.points.x, self.points.y, self.points.z, self.points.i)

            if method == "squared":
                self.stack = fit_ellipse_stack.fit_ellipse_stack_scipy(self.points.x, self.points.y, self.points.z, self.points.i, loss_type="algebraic_squared")

            if method == "eps":
                #self.stack = fit_ellipse_stack_conic.fit_ellipse_stack_scipy(self.points.x, self.points.y, self.points.z, self.points.i, loss_type="algebraic_abs")
                #self.stack = fit_cone_stack_cvxpy.fit_ellipse_stack(self.points.x, self.points.y, self.points.z, self.points.i, norm_type="l2")
                self.stack = fit_ellipse_stack_conic.fit_ellipse_stack_abs(self.points.x, self.points.y, self.points.z, self.points.i)
                #self.stack = fit_ellipse_stack_conic.fit_ellipse_stack_squared(self.points.x, self.points.y, self.points.z, self.points.i)
                #self.stack = fit_ellipse_stack_conic.fit_ellipse_stack(self.points.x, self.points.y, self.points.z, self.points.i)

        print "printing stack"
        for key, value in self.stack.items():
            print key, value

        return self


    ###################################
    # glue data to fitting code

    def evaluate(self):
        """
        count intensity in green channel
        """

        print "shape green channel", self.green_channel.shape

        assert isinstance(self.stack, dict), "type check failed"

        self.evaluation = EvaluationData()

        for (z, e) in self.stack.items():

            intensity = 0
            num_pixels = 0

            assert z == e.cz
            img = self.green_channel[:,:,z]

            rx = e.rx + self.radius_offset
            ry = e.ry + self.radius_offset

            assert rx > 0 and ry > 0

            for x in xrange(img.shape[0]):
                for y in xrange(img.shape[1]):
                    if ((x-e.cx)**2/(rx*rx) + (y-e.cy)**2/(ry*ry) - 1) <= 0:
                        intensity += img[x,y]
                        num_pixels += 1

            self.evaluation.add_layer(num_pixels, intensity)

        return self.evaluation


if __name__ == "__main__":
    # Don't create a new QApplication, it would unhook the Events
    # set by Traits on the existing QApplication. Simply use the
    # '.instance()' method to retrieve the existing one.

    app = QtGui.QApplication.instance()
    container = MainWidget()
    window = QtGui.QMainWindow()
    window.setCentralWidget(container)
    window.show()

    # Start the main event loop.
    app.exec_()

