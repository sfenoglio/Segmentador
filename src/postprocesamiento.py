#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package postprocesamiento
#  Paquete que incluye las funciones necesarias para el post-procesamiento de la predicción obtenida por la red convolucional con el objetivo de mejorar el desempeño de la misma.
#
# Primero se aplica la corrección por canales ‘CC()’ si el umbral ‘umbralCC’ fuera mayor a 0.
# Luego, se aplica el algoritmo k-NN ‘knn()’ si ‘umbralKNN’ fuera mayor a 0, sino se elige la clase definitiva para cada píxel simplemente como la que tenga mayor probabilidad.
# Por último, se aplica la eliminación de pequeñas imperfecciones ‘EPI()’ si ‘umbralEPI’ fuese mayor a 0.

"""
Created on Mon Dec 10 15:11:12 2018

@author: sebastian
"""

import numpy as np
import cv2 as cv
import torch
from sklearn.neighbors import KNeighborsClassifier

#%%
## Transforma la salida lineal de la red convolucional en probabilidades mediante la función softmax de PyTorch.
# 
# Para ello primero convierte la imagen en Tensor de PyTorch y luego reconvierte nuevamente a arreglo de NumPy.
#@param img Arreglo 2D de NumPy con la salida de la red convolucional.
#@return Arreglo de NumPy con probabilidades de pertenencia a cada clase.
def lineal2prob(img):
    imgTensor= np.expand_dims(img, axis= 0)
    imgTensor= torch.Tensor(imgTensor)
    softmax= torch.nn.Softmax2d()
    imgTensor= softmax(imgTensor)
    return imgTensor.numpy()


#%%
## Dada una zona de una imagen, devuelve la clase con más ocurrencias en los vecinos inmediatos.
#
# Para ello, se aplica la operación morfológica de dilatación para obtener los vecinos de la zona indicada y luego se cuentan las clases mediante la función 'unique' de NumPy.
#@param img Arreglo de NumPy de dos dimensiones que contiene en cada píxel la clase a la que éste pertenece.
#@param maskParcial Arreglo de NumPy de dos dimensiones que indica la zona en la que se quiere evaluar los vecinos. Se toman los píxeles mayores a cero.
#@return La clase con más ocurrencias en las inmediaciones de la zona indicada.
def mayorVecino(img, maskParcial):
    ee= cv.getStructuringElement(cv.MORPH_CROSS, (3,3))*255
    borde= cv.dilate(maskParcial,ee) - maskParcial
    imgSoloBorde= cv.bitwise_and(img.astype(np.uint8), borde.astype(np.uint8))
    valores, cantidades= np.unique(imgSoloBorde[imgSoloBorde>0], return_counts=True)
    #ver si esta vacio antes
    if(valores.shape[0]==0):
        #solo fondo hay, devuelvo al maximo dentro de la zona maskParcial
        valores, cantidades= np.unique(img[maskParcial>0], return_counts=True)
        #no deberia haber ningun 0
        valorADevolver= valores[np.argmax(cantidades)]
        if(valorADevolver==0):
            print("Error de mayorVecino()")
        return valorADevolver
    else:
        return valores[np.argmax(cantidades)]


#%%
## Función que descarta las zonas menores al tamaño indicado directamente de la imagen y le asigna la clase que predomine en su vecindad inmediata.
#
# Para ello, genera una máscara para cada clase y busca componentes conexas con el algoritmo 'findContours()' de OpenCV.
# Luego, analiza cada componente y si es menor al tamaño dado 'tamDespreciable', le asigna una clase indefinida número '24'.
# Por último, se detectan las componentes conexas que correspondan a dicha clase indefinida '24' y se le asigna la clase mayoritaria en su vecindad mediante 'mayorVecino()'.
#@param imgFinal Arreglo de NumPy de dos dimensiones que contiene en cada píxel la clase a la que éste pertenece.
#@param tamDespreciable Umbral de tamaño. Se descartan todas las zonas menores a él.
#@return Arreglo de NumPy de dos dimensiones con las zonas menores al umbral de tamaño reemplazadas por su vecino más concurrente.
def EPI(imgFinal, tamDespreciable= 100):
    #1. generar lista de mascaras con promedio de pertenencia
    lista= []
    for i in range(1,24):
        mask= imgFinal==i
        mask= mask.astype(np.uint8)*255
        _, contours, _= cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) #cv.CHAIN_APPROX_SIMPLE
        #1.1 para cada componente conexa
        for cnt in contours:
            #1.2 dibujar mascara de componente conexa
            maskParcial= np.zeros_like(mask)
            cv.drawContours(maskParcial,[cnt],-1,255,cv.FILLED) 
            #1.3 calcular promedio y guardar salidas
            lista.append( maskParcial.copy() )
    
    #2. para cada una
    mascara24= np.zeros_like(imgFinal).astype(np.uint8)
    for i,mask in enumerate(lista):
        #2.1 segun tamaño
        area= np.count_nonzero(mask)
        if(area<tamDespreciable):
            #2.2 cambiar
            mascara24[mask>0]= 255
        
    #3 ver vecinos
    if((mascara24==255).sum()==0):
        #3.1 si no hay zonas despreciables
        return imgFinal
    else:
        #3.2 analizar por componente conexa
        _, contours, _= cv.findContours(mascara24,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            maskParcial= np.zeros_like(mascara24)
            cv.drawContours(maskParcial,[cnt],-1,255,cv.FILLED) 
            #3.3 buscar vecino con mas concurrencia
            vecino= mayorVecino(imgFinal, maskParcial= maskParcial)
            #3.4 asignarlo
            imgFinal[maskParcial==255]= vecino
    
    return imgFinal


#%%
## Postprocesamiento usando el algoritmo de k-NN provisto por la librería scikit-learn.
#
# Se entrena k-NN con los píxeles que tengan una mayor probabilidad a 'umbralKNN' y luego se predice la clase de los restantes.
# Cada píxel tiene 24 valores puesto que posee una probabilidad para cada clase.
#@param img Arreglo NumPy de 24 canales con probabilidades por clase.
#@param umbralKNN Umbral utilizado para determinar los "píxeles confiables".
#@return Arreglo de NumPy de dos dimensiones que contiene en cada píxel la clase a la que éste pertenece.
def knn(img, umbralKNN= .9):   
    #1. inicializar knn
    neigh = KNeighborsClassifier(n_neighbors= 5)
    
    #2. buscar los pixeles "confiables"
    data= img.copy()
    data[data<umbralKNN]= 0
    imgFinal= np.argmax(img, axis=0)
    mask= (np.max(data, axis=0)>umbralKNN).astype(np.uint8)*255 #mascara donde estan los referentes de cada clase

    #3. Entrenamiento de knn
    #3.1 verificar que no este vacia: en ese caso devuelve la imagen con np.argmax
    if((mask==0).all()):
        return imgFinal
    #3.2 buscar las "clases confiables"
    dataAux= img.copy()    
    dataAux[dataAux<umbralKNN]= 0
    clases= np.argwhere(np.sum( dataAux , axis=(1,2))>0)
    X= []
    y= []
    #3.3 generar datos de entrenamiento con su clase
    for i,c in enumerate(clases):
        c= c[0]
        maskClase= (imgFinal==c).astype(np.uint8)*255
        maskClase= np.bitwise_and(maskClase, mask) #combino las mascaras
        agrego= (np.swapaxes(img[:,maskClase.astype(np.bool)],0,1))
        for j in range(agrego.shape[0]):
            X.append(agrego[j])
            y.append(i)
    #3.4 entrenar knn
    print("Entrenando knn...")
    neigh.fit(X, y)
    print("knn entrenado.")
    
    #4. predecir pixeles faltantes
    salida= imgFinal.copy()
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            if(mask[i,j]==0):
                salida[i,j]= clases[neigh.predict([img[:,i,j]])[0]][0] #0 porque es lista de listas no sé porque
    
    #5. devolver
    return salida


#%%
## Funcion utilizada para hacer 0 los canales de las clases que tengan menor probabilidad al umbral y posteriormente corregir las probabilidades.
#
# Se determinan las clases confiables como aquellas que tengan al menos un píxel mayor a 'umbralCC'.
# Luego se hacen 0 los canales de las restantes clases y se recalculan las probabilidades para cada píxel de forma que sumen 1.
#@param img Arreglo de NumPy de cuatro dimensiones correspondiente a la salida de la red convolucional.
#@param umbralCC Umbral utilizado para determinar las "clases confiables".
#@return Arreglo de NumPy de cuatro dimensiones con los canales de las "clases no confiables" llevados a cero.
def CC(dataNP, umbralCC= 0.9):
    #1. identificar clases presentes en cada imagen del batch
    negativos= np.logical_not(np.sum( dataNP>umbralCC , axis=(2,3)).astype(np.bool))
    
    #2. hacer 0 en canales de clases no presentes
    dataNP[negativos,:,:]= 0
    
    #3. corregir para que sumen 1 (simplemente dividir por la suma en cada canal)
    dataNP /= np.stack([np.sum(dataNP, axis= 1) for i in range(24)], axis=1)
    
    #4. devolver
    return dataNP


#%%
## Función que integra a las demás para cumplir el objetivo del paquete.
#@param data Arreglo 4D de NumPy correspondiente a la salida de la red convolucional.
#@param umbralCC Umbral utilizado para la determinación de "clases confiables" para la corrección por canales. Si es 0 no se hace.
#@param umbralKNN  Umbral utilizado para la determinación de "píxeles confiables" para la corrección mediante el algoritmo de k-nn. Por defecto es 0 y no se hace.
#@param umbralEPI Umbral que se utiliza en la corrección de pequeñas imperfecciones. Por defecto es 100. Si es 0, no se aplica EPI.
#@return Arreglo de NumPy de dos dimensiones que contiene en cada píxel la clase a la que éste pertenece.
def postprocesar(data, umbralCC= 0.9, umbralKNN= 0, umbralEPI= 100):
    #1. Transformar a probabilidades mediante la función softmax y luego a formato arreglo de numpy
    img= lineal2prob(data)
    
    #2. Corregir por canales según umbral (si es distinto a 0)
    if(umbralCC!=0):
        img= CC(img, umbralCC)
    img= img[0]
    
    #3. Corregir mediante k-nn (si el umbral es distinto a 0)
    if(umbralKNN!=0):
        img= knn(img, umbralKNN)
    else:
        # sino se elige directamente la clase de máxima probabilidad por píxel
        img= np.argmax(img, axis=0)
    #4. Corregir mediante corrección de pequeñas imperfecciones (según booleano)
    if(umbralEPI>0):
        img= EPI(img, umbralEPI)
    
    #5. Devolver
    return img
