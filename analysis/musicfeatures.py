#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import librosa
import matplotlib.pyplot as plt 
from sklearn import decomposition
from sklearn.feature_selection import SelectPercentile, f_classif
from descriptors import * 

######################################################################################
#Features extraction
######################################################################################
class Features:
    """Class for music features extraction"""
    def __init__(self,filename,category,piece_len=60):
        """
        Load file from path *filename* using librosa library.
        You have to set *category* of each file and you can define its length
        *piece_len*.
        """
        self.y, self.sr = librosa.load(filename,duration=piece_len+1)
        self.piece_len = piece_len
        self.category = category
        self.features = []
        self.winflag = False
    def _featadd(self,signal,function,params=0):
        try:
            if params:
                return [x for x in function(signal,**params)]
            else:
                return ([x for x in function(signal)])
        except TypeError:
            if params:
                return [function(signal,**params)]
            else:
                return [function(signal)]
    def add_features(self,function, params = 0):
        """It adds all features calculated from *function* with *params*
        for each window (if windowing was done)."""
        if self.winflag == False:
            self.features.extend(self._featadd(self.y,function,params))
        else:
            for sig in self.cutted:
                self.features.extend(self._featadd(sig,function,params))
    def add_winbased_features(self,function, params = 0, comparefun = np.mean):
        """It applies function *comparefun* for features calculated from 
        *function* with *params* for windows (if windowing was done)."""
        assert self.winflag==True
        feat_ = []
        for sig in self.cutted:
            feat_.append(self._featadd(sig,function,params))
        _cf = comparefun(np.array(feat_),axis=0)
        try:
            self.features.extend(list(_cf))
        except TypeError:
            self.features.extend([_cf])
    def windowing(self, win_len,win_step):
        """It cuts music piece for windows of length *win_len* and step
        *win_step* given in seconds"""
        self.winflag = True
        beginnings = np.arange(0,self.piece_len*self.sr,win_len*self.sr-win_step*self.sr)[:-1]
        self.cutted = np.zeros((len(beginnings),win_len*self.sr))
        for e,b in enumerate(beginnings):
            self.cutted[e,:] = self.y[b:b+win_len*self.sr]
    @property
    def example(self):
        "Returns tuple with all calculated features and category"
        return np.array(self.features),self.category

######################################################################################
#Postprocessing classes and functions
######################################################################################

class Num:
    "Class for numerizing text categories"
    def numerize(self,classes,dc=0):
        "It changes string label into numbers - if *dc* is a dict it uses given coding"
        if not dc:
            self.new_classes = []
            only_one = set()
            for x in classes:
                only_one.add(x)
            numclass = range(len(only_one))
            self.num_dc = dict()
            self.denum_dc = dict()
            for k,n in zip(only_one,numclass):
                self.num_dc[k]=n
                self.denum_dc[n]=k
            for c in classes:
                self.new_classes.append(self.num_dc[c])
        else:
            self.num_dc = dc
            self.new_classes = [dc[c] for c in classes]
            self.denum_dc = dict()
            for k,v in dc.items():
                self.denum_dc[v]=k
        return self.new_classes
    def denumerize(self, values):
        "From list of values it gives you actual labels"
        return [self.denum_dc[v] for v in values]

def normalize(features):
    "Normalize features from 0 to 1"
    mxs = np.max(features,axis=0)
    mns = np.min(features,axis=0)
    return (features - mns)/(mxs-mns)

def standarize(features):
    "Standarize features: (x_i-mu)/std"
    mean = np.mean(features,axis=0)
    std = np.std(features,axis=0)
    return (features - mean)/std

def pca(features, var_expl = 0.98, n_com=None):
    """
    Returns features with dimension reduced by PCA method
    implemented in scikit learn (sklearn) module. Number of components
    is matched based on explained variance ratio - *var_expl* or 
    can be set by hand as *n_com*.
    """
    pca = decomposition.PCA()
    pca.fit(features)
    if not n_com:
        for p in xrange(1,features.shape[0]):
            if np.sum(pca.explained_variance_ratio_[:p])>=var_expl:
                n_com = p
                break
    pca = decomposition.PCA(n_components=n_com)
    pca.fit(features)
    features = pca.transform(features)
    return features

def select_feat(X,y,percentile=20):
    "Select best 20 % of features using Anova F-value - *f_classif* from scikit.learn"
    selector = SelectPercentile(f_classif, percentile=percentile)
    selector.fit(X, y)
    return  selector.transform(X)

if __name__ == '__main__':
    mus = Features('music/GoHome.mp3','sad',piece_len=20)
    params = {'fs':mus.sr, 'n_fft':4096, 'hop_len':64}
    mus.windowing(5,1)
    mus.add_winbased_features(rms)
    mus.add_winbased_features(chromagram_feat, params)
    print mus.example