#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package segmentador
#  Archivo que incluye a los demás módulos para la integración y ejecución de la herramienta.
#
# Primero define los parámetros mediante 'interfaz.py' para luego carga las imágenes de entrada con 'interfaz.leerImagenes()'.
# Para cada imagen, conecta los módulos 'preprocesamiento', 'separacion' y 'postprocesamiento' pasando la salida de uno a la entrada del siguiente.
# A continuación, guarda las imágenes de salida utilizando la funcion 'interfaz.guardarImagenes()'.
# El proceso se repite si hubiese más imágenes para finalmente guardar los parámetros por defecto mediante 'interfaz.saveDefaultArgs()' si así se pidiese.

"""
Created on Tue Jan 29 20:20:34 2019

@author: sebastian
"""

import cv2 as cv
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import time
from src.interfaz import *


#%% -------------------------------------------------------------------------------------------------------------------
#                                                       MAIN
#   -------------------------------------------------------------------------------------------------------------------

imgs, paths_out= leerImagenes(args.path_img, args.path_out)

#para cada imagen
for i in range(len(imgs)):
    print("Procesando imagen "+str(i+1)+ " de un total de "+str(len(imgs))+"...")
    img= imgs[i]
    path_out= paths_out[i]
    
    
    #controlar que no esté vacia
    if(type(img)!= np.ndarray):
        print("Error en la lectura la imagen.")
        continue
        
    
    #crear carpeta
    try:
        from shutil import rmtree
        rmtree(path_out)
    except:
        time.sleep(0.1)
    try:
        os.mkdir(path_out)
    except:
        print("No se puede crear la carpeta de salida: "+path_out)
        continue
        
    
    #preprocesar
    rois, rois_mask= pre.preprocesar(img= img,
                                    tamCuadFondo= args.tamCuadFondo, 
                                    tamCuadUmbral= args.tamCuadUmbral, 
                                    maxTamAgujero= args.maxTamAgujero, 
                                    eeSize= args.eeSize, 
                                    umbralSegm= args.umbralSegm, 
                                    umbralArea= args.umbralArea, 
                                    umbralCH= args.umbralCH, 
                                    TilesGridSize= args.TilesGridSize, 
                                    ClipLimit= args.ClipLimit)

    if(args.guardarPreproc):
        for j,roi in enumerate(rois):
            cv.imwrite(path_out+str(j).zfill(2)+".png", roi)


    #separar
    rois, separadas= sep.separar(rois,
                           imgs_mask= rois_mask if(args.aplicarMaskPreproc) else [], #para que aplique o no la mascara de preprocesamiento
                           tamImagen= (args.tamImagenSeparar,args.tamImagenSeparar),
                           std= args.adecStd,
                           blur= args.adecBlur,
                           invertirImg= args.invertirImg)
    
    
    #postprocesar cada imagen
    for j in range(len(separadas)):
        #postprocesar
        separadas[j]= post.postprocesar(data= separadas[j],
                                     umbralCC= args.umbralCC, 
                                     umbralKNN= args.umbralKNN, 
                                     umbralEPI= args.umbralEPI)
    
    
    #guardar
    guardarImagenes(rois, separadas, path_out)
    cv.imwrite(path_out+"original.png", img)
    
    print("Imagen "+str(i+1)+ " de un total de "+str(len(imgs))+" procesada.")
    
    
#%% guardar parametros por defecto al final de la ejecucion
if(args.saveDefaultArgs):
    guardarDefaultArgs()
