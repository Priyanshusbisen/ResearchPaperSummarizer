import torch
import os
import urllib.request
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.manifold import TSNE
import numpy as np
import requests
import zipfile
import io
import chakin
import json
import os

class WordEmbeddings():
    """Class to download pretrained word embeddings"""
    def __init__(self, data):
        self._chaking_index = 16
        self._num_dimentions = 300
        self._subfolder_name = 'glove.240B.300d'
        self._data_folder = 'embeddings'
        self._zip_path = os.path_join(self._data_folder,f'{self._subfolder_name}')
        self._unzip_folder = os.path.join(self._data_folder,self._subfolder_name)
        self._glove_file_name = os.path.join(self._unzip_folder,f'{self._subfolder_name}.txt')
        
        if not os.path.exists(self._unzip_folder) and not os.path.exists(self._zip_path):
            self._retrieve()
        if not os.path.exists(self._unzip_folder):
            self._extract()
        self.model = None
        
    def _retrieve(self):    
        # GloVe by Stanford is licensed Apache 2.0: 
        #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
        #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
        #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
        print("Downloading embeddings to '{}'".format(self._zip_path))
        chakin.download(number=self._chaking_index, save_dir='./{}'.format(self._data_folder))

    def _extract(self):
        # Extract the zip file
        with zipfile.ZipFile(self._zip_path, 'r') as zip_ref:
            zip_ref.extractall(self._unzip_folder)

    def _load_glove(self):
        #Load the glove model
        with open(f'{self._glove_file_name}', 'r') as data:
            glove_file = data

    def fit(self, data):
        pass
        

        





