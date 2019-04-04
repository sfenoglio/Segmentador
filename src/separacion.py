#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package separacion
#  Paquete que incluye la clase separacion, necesaria para separar un solapamiento de cromosomas mediante la aplicación de una red convolucional.
# Se inicializa la arquitectura de la red, se cargan sus parámetros entrenados y se manda a la GPU si se pidiera y pudiera en ‘__init__()’. 
# Luego se incluyen métodos para asemejar la imagen de entrada a los datos sintéticos generados ‘procesarImg()’ si así se quisiese, para redimensionar la imagen ‘desplazarResize()’ y para pasar la imagen por la red convolucional ‘inferir()’. Por último, el método 'separar()' combina todas las funciones anteriores. 
"""
Created on Wed Jan 30 22:50:29 2019

@author: sebastian
"""

#postprocesamiento
try:
    import torch
except:
    print("No se encuentra la librería pytorch.")
    quit()

import numpy as np
import cv2 as cv
    

class separacion():
   #%% 
    ## Constructor de la clase, encargado de cargar los parámetros entrenados de un modelo dado.
    #
    # Tener en cuenta que el modelo debe estar guardado en el formato que se indica en el parámetro. 
    # Esto es por compatibilidad con el proyecto utilizado para el entrenamiento de la red convolucional.
    #@param path Path del archivo que contiene los parámetros del modelo. Debe estar en un diccionario con la clave "state_dict" y guardado serialmente (idealmente mediante la función de pytorch).
    #@param model Modelo de pytorch al cual se le cargarán los parámetros entrenados.
    #@return Modelo de pytorch con los parámetros cargados. 
    def __init__(self, path, model, gpu=True):
        self.model= model
        self.gpu= gpu
        
        #importar modelo entrenado
        print("Cargando estado anterior de modelo...")
        try:
            state= torch.load(path)
            self.model.load_state_dict(state["state_dict"])
        except:
            print("El modelo entrenado indicado no existe o hubo un error al cargarlo.")
            quit()
            
        if(self.gpu):
            try:
                self.model= self.model.cuda()
            except:
                print("No se encuentra GPU.")
                quit()
        
        print("Estado anterior de modelo cargado.")
        

    #%%
    ## Función que aplica histogram matching con una gaussiana de media 128 y desvío indicado.
    #
    #
    #@param source Imagen o píxeles a los que se aplicará.
    #@param sigma Desvío de la gaussiana que se utilizará.
    #@return Imagen o píxeles corregidos según la gaussiana.
    def histGauss(self, source, sigma= 60):
        oldshape = source.shape
        source = source.ravel()
    
        from scipy import signal
        t_values= np.arange(0,254).astype(np.uint8)
        t_counts = np.append(signal.gaussian(250, std= sigma), np.zeros((4)))
    
        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
    
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
    
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
        return interp_t_values[bin_idx].reshape(oldshape).astype(np.uint8)
    
    #%%
    ## Función que agranda la imagen que recibe al tamaño especificado, dejando en el centro la primera.
    #
    # Si la imagen fuera mayor al tamaño deseado, éste se duplica y se intenta nuevamente.
    #@param img Imagen a agrandar en formato de arreglo numpy de dos dimensiones.
    #@param tamImagenSalida Tupla que indica el tamaño de salida deseado.
    #@param fondo Indica el color del fondo para rellenar la imagen agrandada.
    #@return Imagen agrandada en formato de arreglo numpy de dos dimensiones.
    def desplazarResize(self,img,tamImagenSalida,fondo=255):
        #1. inicializo
        salida= (fondo*np.ones(tamImagenSalida)).astype(np.uint8)
        H,W= img.shape
        
        #2. calculo desplazamientos
        despl_h= (tamImagenSalida[0]-H)//2
        despl_w= (tamImagenSalida[1]-W)//2
        
        #3. los aplico
        if(H<=tamImagenSalida[0] and W<tamImagenSalida[1]):
            salida[despl_h:despl_h+H, despl_w:despl_w+W]= img.copy()
        else:
            return self.desplazarResize(img, (tamImagenSalida[0]*2,tamImagenSalida[1]*2), fondo)
    
        #4. devuelvo
        return salida
    
    #%%
    ## Procesa una imagen de un cromosoma calculando su inversa, aplicando histogram matching y un filtro gaussiano de 3x3.
    #
    # 
    #@param img Imagen a segmentar en formato de arreglo numpy de dos dimensiones.
    #@param std Desvío utilizado para aplicar histogram matching con una gaussiana.
    #@param blur Desvío utilizado en el filtro gaussiano.
    #@param invertirImg Booleano que indica si se calcula la inversa de la imagen o no.
    #@param fondo Indica el color del fondo para excluirlo del calculo del histogram matching.
    #@param Imagen de dos dimensiones indicando en cada píxel el número de clase.
    def procesarImg(self, img, std= 60, blur=0.7, invertirImg= True, fondo= 255):
        if(std>0):
            img[img!=fondo]= self.histGauss(img[img!=fondo], std)
        if(blur>0):
            img= cv.GaussianBlur(img, (3,3), blur)
        if(invertirImg):
            img= 255 - img
        return img


    #%%
    ## Dada una imagen, devuelve su segmentación mediante el pasaje de la misma por la red convolucional.
    #
    # También se encarga del pasaje de numpy a tensor de pytorch para la inferencia y del pasaje inverso para la devolución del resultado.
    #@param img Imagen a segmentar en formato de arreglo numpy de dos dimensiones.
    #@return Imagen de dos dimensiones indicando en cada píxel el número de clase.
    def inferir(self,img):
        #crear torch tensor en 4D y llevar a [0-1]
        t_img= np.expand_dims(np.expand_dims(img/255,0),0).astype(np.float32)
        t_img= torch.from_numpy(t_img)
        
        with torch.no_grad():
            if(self.gpu):
                t_img= t_img.cuda()
            #forward
            output= self.model(t_img)
        
        #volver a cpu
        if(self.gpu):
            predicted = output.data.cpu()
        else:
            predicted = output.data
        
        #pasar a numpy y volver a 3D
        return predicted.numpy()[0]


    #%% 
    ## Función integradora de las demás del paquete.
    #
    # No solo conecta las demás funciones, sino que también se encarga de manejar múltiples inferencias sin tener que recargar el modelo.
    #@param imgs Lista de imágenes a segmentar.
    #@param imgs_mask Lista de máscaras de las imágenes a segmentar. Si la lista tiene longitud distinta a imgs, no se aplica.
    #@param tamImagen Tupla que indica el tamaño al que se agrandará la imagen antes de pasar por la red convolucional.
    #@param std Desvío utilizado para histogram matching. Si es 0 no se aplica.
    #@param blur Desvío utilizado para aplicar filtro gaussiano de 3x3. Si es 0 no se aplica.
    #@return Tupla con dos listas. Una de la imagen original ampliada y otra de imágenes de dos dimensiones indicando en cada píxel el número de clase.
    def separar(self,imgs, imgs_mask= [], tamImagen= (256,256), std= 60, blur=0.7, invertirImg= True):
        
        #para cada imagen
        salida= []
        salida_imgs= []
        self.model.eval()
        for i in range(len(imgs)):
            cromo= self.desplazarResize(imgs[i], tamImagen, fondo= 255 if invertirImg else 0)
            cromo= self.procesarImg(cromo,std,blur,invertirImg, fondo= 255 if invertirImg else 0)
            salida_imgs.append(cromo)
            inferencia= self.inferir(cromo)
            
            if(len(imgs)==len(imgs_mask)):
                cromo_mask= self.desplazarResize(imgs_mask[i], tamImagen, fondo= 0) #mascara siempre es 0
                #hacer [1 0 ....] en todos los canales los pixeles que son 255
                #canal 0 es 1 y el resto 0
                inferencia [0, cromo_mask!=255]= 10000
                inferencia [1:, cromo_mask!=255]= 0
            
            salida.append(inferencia.copy())
    
        #devuelvo
        return (salida_imgs, salida)
