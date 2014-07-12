#!/usr/bin/python
#-*- coding: utf-8 -*- 

__version__ = '0.7'
__author__ = 'Dominik Krzeminski'

import sys
from PyQt4 import QtGui, QtCore
from PyQt4.phonon import Phonon

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt
import random 
import numpy
from analysis.descriptors import * 
from analysis.musicfeatures import Features, Num, normalize

CONFIG = {'model': 'model_allb'}# model_tri1
params = {'n_fft':4096, 'hop_len':64, 'func': np.mean}

def depickle(model_name):
    """
    From given pickle file it receives info about:
    classifier, normalization params, features info, labels coding 
    """
    import cPickle

    with open('%s.pkl'%model_name,'rb') as f:
        model = cPickle.load(f)

    clf = model['classifier']
    normin = model['norm']['min']
    normax = model['norm']['max']
    featinfo = model['featinfo']
    coding = model['coding']
    return clf, normin, normax, featinfo, coding

def calculate_features(path,piece_len=30):
    """
    It return features and unknown class 'x' for music piece from *path*
    Set of features:  rms, hoc, beats, chromagram, tempo, spectral centroids
    """
    try:
        musicfeat = Features(path,'x',piece_len=piece_len)
        params.update({'fs': musicfeat.sr})
        musicfeat.windowing(10,1) 
        musicfeat.add_winbased_features(rms)
        musicfeat.add_winbased_features(simple_hoc)
        musicfeat.add_winbased_features(beats,params)
        musicfeat.add_winbased_features(chromagram_feat, params)
        musicfeat.add_winbased_features(tempo,params)
        musicfeat.add_winbased_features(spectral_centroids,params)
        feats,clas = musicfeat.example
    except Exception as e:
        print e
        feats,clas = 0,0
    return feats, clas

class MusicEmoReco(QtGui.QMainWindow):
    "Main application for music emotion recognition"
    def __init__(self, parent=None):
        super(MusicEmoReco, self).__init__(parent)
        self.setWindowTitle('Music Emotion Recognition')
        #music
        self.mediaObject = Phonon.MediaObject(self)
        self.audioOutput = Phonon.AudioOutput(Phonon.MusicCategory, self)
        Phonon.createPath(self.mediaObject, self.audioOutput)
        #self.mediaObject.stateChanged.connect(self.handleStateChanged)

        central_widget1=self.initMenu()
        central_widget2=self.initPlot()
        mainWidget = QtGui.QWidget(self)
        mainlay = QtGui.QVBoxLayout()
        mainlay.addWidget(central_widget1)
        mainlay.addWidget(central_widget2)
        mainWidget.setLayout(mainlay)
        self.setCentralWidget(mainWidget)
        self.load_model()

    def initMenu(self):
        "The most important buttons and fields are here created - it returns a QWidget object with them."
        cWidget = QtGui.QWidget(self)
        self.play_button = QtGui.QPushButton('Play')#Choose_button()
        self.play_button.clicked.connect(self.playAction)
        self.choose_button = QtGui.QPushButton('Choose file')
        self.choose_button.resize(100,50)
        self.choose_button.clicked.connect(self.handleButton)
        self.path_le= QtGui.QLineEdit(self)
        self.statusLabel = QtGui.QLabel("Ready!")
        self.statusLabel.setAlignment(QtCore.Qt.AlignRight)
        grid = QtGui.QGridLayout(cWidget)
        grid.setSpacing(0)

        grid.addWidget(self.choose_button,0,0)
        grid.addWidget(self.path_le,0,1)
        grid.addWidget(self.play_button,1,1)
        grid.addWidget(self.statusLabel,3,1)
        return cWidget

    def initPlot(self):
        "Return QWidget with canvas to plot on it."
        cWidget = QtGui.QWidget(self)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QtGui.QVBoxLayout(cWidget)
        self.plot_button = QtGui.QPushButton('Check emotion')
        self.plot_button.clicked.connect(self.make_features)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.canvas)
        cWidget.setLayout(layout)
        return cWidget

    def plot_reco(self,predictions):
        "Plots results of classification"
        ax = self.figure.add_subplot(111)
        ind = np.arange(len(self.clf.classes_))
        width = 0.4 
        predictions*=100
        ax.hold(False)
        r = ax.bar(ind, predictions, width, color='g')
        ax.set_ylabel('Probability [%]')
        ax.set_xlabel('Emotion')
        ax.set_xticks(ind+width)
        ax.set_ylim([0,100])
        ax.set_xticklabels(self.clf.classes_)
        self.canvas.draw()

    def handleStateChanged(self, newstate, oldstate):
        "Choosing a file"
        if newstate == Phonon.PlayingState:
            self.setText('Stop')
        elif newstate == Phonon.StoppedState:
            self.setText('Choose File')
        elif newstate == Phonon.ErrorState:
            source = self.mediaObject.currentSource().fileName()
            print 'ERROR: could not play:', source.toLocal8Bit().data()
    def playAction(self):
        "After music button pressed it plays or stops music"
        if self.mediaObject.state() == Phonon.StoppedState:
            self.mediaObject.play()
            self.play_button.setText('Stop')
        elif self.mediaObject.state() == Phonon.PlayingState:
            self.mediaObject.stop()
            self.play_button.setText('Play')          
    def handleButton(self):
        "Controling states of music piece"
        if self.mediaObject.state() == Phonon.PlayingState:
            self.mediaObject.stop()
        path = QtGui.QFileDialog.getOpenFileName(self, self.choose_button.text())
        if path:
            self.path_le.setText(path)
            self.mediaObject.setCurrentSource(Phonon.MediaSource(path))
    def ups(self,mes):
        "It shows dialog window with error message *mes*"
        QtGui.QMessageBox(QtGui.QMessageBox.Warning,
            "Error", mes,
            QtGui.QMessageBox.NoButton, self).show()

    def load_model(self):
        "It loads a file with model saved as a dictionary in python cPickle"
        try:
            self.clf, self.normin, self.normax, featinfo, coding = depickle(CONFIG['model'])
        except Exception as e:
            raise e

    def make_features(self):
        "Communication with FeatureThread"
        path = str(self.path_le.text())
        self.thread = FeatureThread(path)
        self.thread.result.connect(self.catch_values)
        self.thread.status.connect(self.check_thread)
        self.thread.start()

    def check_thread(self,val):
        "Checking if thread sstatus - if it is done it starts classify"
        if val==True:
            self.statusLabel.setText('Ready')
            self.plot_button.setDisabled(False)
            self.classify()
        else:
            self.statusLabel.setText('Calculating...')
            self.plot_button.setDisabled(True)
    def catch_values(self, val):
        "Catch results from thread"
        self.feats, self.clas = val

    def classify(self):
        "If features was calculated it plots bars, otherwise it returns error window"
        if type(self.feats)!=int and self.clas!=0:
            self.feats = (self.feats - self.normin)/(self.normax - self.normin)
            self.plot_reco(self.clf.predict_proba(self.feats)[0])
        else:
            self.ups('Bad file format!')

class FeatureThread(QtCore.QThread):
    "Calculating features for classification"
    result = QtCore.pyqtSignal(tuple)
    status = QtCore.pyqtSignal(bool)
    def __init__(self,path):
        super(FeatureThread, self).__init__()
        self.path = path
    def run(self):
        self.status.emit(False)
        feats,clas = calculate_features(self.path)
        self.result.emit((feats,clas))
        self.status.emit(True)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = MusicEmoReco()
    main.show()

    sys.exit(app.exec_())
