#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:15:35 2018

@author: javier
"""

import numpy as np
path= '/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/allWeights/'
convo = np.load('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/laputasalidadelaconvo.npy')
convor = convo.reshape ((1,150))
for i in range(120):
    File  = path + 'Output'+ str(i) + '.npy'
    outputFlatten = np.load(File)
    substraction = convor - outputFlatten
    print(sum(sum(substraction)))
    
    
    

