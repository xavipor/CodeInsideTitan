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
for m in range(120):
    for n in range(120):
        File  = path + 'Output'+ str(m) +'_'+str(n)+ '.npy'
        outputFlatten = np.load(File)
        substraction = convor - outputFlatten
        
        if outputFlatten[0][1]==convor[0][1]:
            print (m,n,sum(sum(substraction)))


    
    

