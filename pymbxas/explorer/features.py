#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:19:59 2024

@author: roncofaber
"""

# numpy stuff
import numpy as np

# sklear or die
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# random
from collections import defaultdict

#%%

class PFA(object):
    def __init__(self, diff_n_features = 2, q=None, explained_var = 0.95,
                 rseed=42):
        self.q = q
        self.diff_n_features = diff_n_features
        self.explained_var = explained_var
        self._rseed = rseed
        
    def fit(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(random_state=self._rseed).fit(X)
        
        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
            for i,j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break
                    
        A_q = pca.components_.T[:,:q]
        
        clusternumber = min([q + self.diff_n_features, X.shape[1]])
        
        kmeans = KMeans(n_clusters= clusternumber, random_state=self._rseed).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        
        return
        
    def fit_transform(self,X):    
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(random_state=self._rseed).fit(X)
        
        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
            for i,j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break
                    
        A_q = pca.components_.T[:,:q]
        
        clusternumber = min([q + self.diff_n_features, X.shape[1]])
        
        kmeans = KMeans(n_clusters=clusternumber, random_state=self._rseed).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        
        return X[:, self.indices_]
    
    def transform(self, X):
        return X[:, self.indices_]
    
    @classmethod
    def filter_data(cls, datas, n_features=1, explained_var=0.975, rseed=42):
        out_data = []
        
        for data in datas:
            # Concatenate data along the feature axis
            data = abs(np.concatenate(data, axis=1).T)
            
            # make pfa obj
            pfa = cls(diff_n_features=n_features, explained_var=explained_var,
                      rseed=rseed)
            
            filtered_features = pfa.fit_transform(data)
            
            out_data.append(filtered_features)
        
        return np.concatenate(out_data, axis=1)