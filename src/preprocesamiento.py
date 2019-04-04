#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package preprocesamiento
#  Paquete que incluye las funciones necesarias para el pre-procesamiento de la imagen de entrada y la extracción de los cromosomas del fondo.
#
# Primero se verifica que el fondo sea negro, de lo contrario calcula la inversa de la imagen de entrada 'img', si 'tamCuadFondo' es mayor a 0. 
# Luego, en este orden, se van aplicando las funciones del paquete para realzar la imagen 'realceImagen()', binarizarla con 'umbralAdaptado', eliminar residuos mediante 'eliminarResiduos()' rellenarle los agujeros con 'rellenoAgujeros'. 
# Al resultado, se le analiza las componentes conexas para la eliminación de residuos mediante 'compConexas'. 
# Por último, se devuelven los objetos segmentados en forma de lista con su correspondiente máscara. 

"""
Created on Wed Aug  1 19:26:14 2018

@author: sebastian
"""

import numpy as np
import cv2 as cv
from skimage import morphology as morph


#%% PREPROCESAR
## Función que integra a las demás para cumplir el objetivo del paquete.
#
# Primero se verifica que el fondo sea negro, de lo contrario calcula la inversa de la imagen de entrada 'img', si 'tamCuadFondo' es mayor a 0. 
# Luego, en este orden, se van aplicando las funciones del paquete para realzar la imagen 'realceImagen()', binarizarla con 'umbralAdaptado', eliminar residuos mediante 'eliminarResiduos()' rellenarle los agujeros con 'rellenoAgujeros'. 
# Al resultado, se le analiza las componentes conexas para la eliminación de residuos mediante 'compConexas'. 
# Por último, se devuelven los objetos segmentados en forma de lista con su correspondiente máscara.
#@param img Imagen en escala de grises.
#@param tamCuadFondo Tamaño del cuadrado del fondo que se toma de las esquinas. Si es 0, no se verifica el fondo.
#@param tamCuadUmbral Lista que contiene los tamaños de la ventana cuadrada que se utiliza en el umbral adaptado. Si es 0, calcula el umbral de Otsu sobre toda la imagen.
#@param maxTamAgujero Tamaño máximo que puede tener un agujero interno a un cromosoma para que se rellene. Cuando es mayor, no se rellena.
#@param eeSize Tamaño del elemento estructurante que es utilizado para la eliminación de los residuos pequeños.
#@param umbralSegm Umbral que determina el tamaño máximo que puede tener el segmento en contacto con el borde de la imagen de un elemento. Si es mayor, se elimina. Si es 0 elimina todos los objetos que están en el borde.
#@param umbralArea Umbral que determina el tamaño máximo que puede tener un elemento para no ser considerado un posible residuo. Si es mayor, se utiliza el criterio del convex hull.
#@param umbralCH Umbral que determina la máxima proporción entre el área de un objeto y el área de su convex hull para no ser considerado un residuo. Si es mayor, se elimina.
#@param TilesGridSize Tamaño de las ventanas cuadradas que se aplican para CLAHE.
#@param ClipLimit Límite para el contraste utilizado en CLAHE.
#@return Tupla de listas. En la primera, cada elemento es una imagen de un objeto segmentado, mientras que en la segunta está su correspondiente máscara que indica con 255 los píxeles donde está el cromosoma y con 0 donde es fondo.
def preprocesar(img,  tamCuadFondo=10, tamCuadUmbral= [0,100] , maxTamAgujero= 10, eeSize= 7, \
                umbralSegm= 30, umbralArea= 4000, umbralCH= 0.8, TilesGridSize=8, \
                ClipLimit= 40): 
    data= img.copy()
    #Integra las demas funciones
    # 1. Que el fondo sea negro
    if(tamCuadFondo>0):
        if(esNegroFondo(img,tamCuadFondo)):
            data= 255 - data
    # 2. Realzado de Imagen
    data= realceImagen(data, TilesGridSize, ClipLimit)
    # 3. Umbral para binarizar
    mask= umbralAdaptado(data,tamCuadUmbral)
    # 4. Eliminación de residuos pequeños
    mask= eliminarResiduos(mask,eeSize)
    # 5. Relleno de agujeros
    maskRellena= rellenoAgujeros(mask, maxTamAgujero)
    # 6. Analisis de componentes conexas
    componentes= compConexas(maskRellena, umbralSegm, umbralArea, umbralCH)
    # 7. Devolver vector de objetos segmentados + normalizacion
    return dividirROIs(data,maskRellena,componentes)
    

#%% FONDO NEGRO
## Verifica si el fondo de la imagen es negro o no tomando ROIs de las esquinas de la imagen.
#
#@param img Imagen en escala de grises en formato de arreglo de NumPy de dos dimensiones.
#@param tamCuadFondo Tamaño del cuadrado del fondo que se toma de las esquinas.
#@return Verdadero si el promedio de los valores de las ROIs es más cercano a 0, de lo contrario Falso.
def esNegroFondo(img,tamCuadFondo=10):
    #cuadrado es el tamanio del cuadrado para tomar de cada esquina de la imagen
    H,W = img.shape
    
    c= tamCuadFondo
    suma= np.sum([ img[0:c,0:c] , img[H-c:H,0:c] , img[0:c,W-c:W] ,img[H-c:H,W-c:W] ])
    promedio= suma / (4*tamCuadFondo**2)
    
    if(promedio<128):
        return True #es negro el fondo
    else:
        return False


#%% realzado de imagen
## Función que realza una imagen en escala de grises aplicando CLAHE y luego normalizando en el rango [0-255].
#
# Primero se aplica Constrained Limited Adaptive Histogram Equalization (CLAHE), que aplica la ecualización por ROIs limitando la amplificación del contraste. Luego, se normaliza en el rango [0-255] mediante transformaciones afines. Ambas operaciones se realizan con las funciones provistas por OpenCV.
#@param img Imagen en escala de grises.
#@param TilesGridSize Tamaño de las ventanas cuadradas que se aplican para CLAHE.
#@param ClipLimit Límite para el contraste utilizado en CLAHE.
#@return Imagen en escala de grises realzada.
def realceImagen(img, TilesGridSize=8, ClipLimit= 40):
    eqCLAHE= img.copy()
    if(TilesGridSize>0 and ClipLimit>0):
        #CLAHE
        clahe= cv.createCLAHE()
        clahe.setTilesGridSize((TilesGridSize,TilesGridSize))
        clahe.setClipLimit(ClipLimit)
        eqCLAHE= clahe.apply(img)
    #normalizado en [0-255]
    normalizada= np.zeros_like(eqCLAHE)
    normalizada= cv.normalize(eqCLAHE,normalizada,255,0,cv.NORM_MINMAX)
    return normalizada


#%% umbral adaptativo: otsu en cuadrados de tamanio tam en la imagen img.
## Función que segmenta mediante la combinación de umbrales adaptados de Otsu de distintos tamaños de ventana.
#
# La segmentación de la imagen se realiza calculando el umbral de Otsu por ventanas usando la función provista por OpenCV para tal fin y luego se interpola para llevar los umbrales al tamaño de la imagen de entrada. Luego, aplica el umbral a cada píxel. Por último, combina los resultados de cada tamaño de ventana utilizado mediante operaciones OR bit a bit entre ellas.
#@param img Imagen en escala de grises a segmentar en formato de arreglo de NumPy de dos dimensiones.
#@param tamCuadUmbral Lista que contiene los tamaños de la ventana cuadrada que se utiliza en el umbral adaptado. Si es 0, calcula el umbral de Otsu sobre toda la imagen.
#@return Imagen binaria resultante de aplicar los umbrales correspondiente a cada tamaño. El resultado es un OR bit a bit entre cada máscara obtenida.
def umbralAdaptado(img,tamCuadUmbral=[0,100]):
    H,W= img.shape
    salida= np.zeros_like(img)
    
    for tam in tamCuadUmbral:
        if(tam<=0):
            #otsu normal
            _, resultadoOtsu= cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            salida= cv.bitwise_or(salida, resultadoOtsu)
            continue
            
        #calculo de tamanio de matriz que contendra los umbrales de Otsu
        HOtsu= H // tam #+ (1 if(H%tam>0) else 0)
        WOtsu= W // tam #+ (1 if(W%tam>0) else 0)
        #inicializacion matriz
        MOtsu= np.zeros((HOtsu,WOtsu), dtype=np.uint8)
        
        for i in range(HOtsu):
            for j in range(WOtsu):
                #para cada cuadrado subimagen
                subImg= img[i*tam:(i+1)*tam-1 , j*tam:(j+1)*tam-1]
                
                #calculo de umbral de Otsu
                MOtsu[i,j],_= cv.threshold(subImg,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                MOtsu[i,j]= np.max([1,MOtsu[i,j]])
        
        #version2
        umbrales= np.zeros_like(img)
        umbrales= cv.resize(MOtsu,(W,H),interpolation=cv.INTER_LINEAR) 
        
        #aplicar umbral a cada pixel
        resultadoAdaptado= np.uint8((img<umbrales)*255)
        
        salida= cv.bitwise_or(resultadoAdaptado, salida)

    return salida


#%% Eliminación residuos pequeños
## Función que elimina los pequeños residuos mediante operaciones morfológicas.
#
# Primero se realiza una erosión con un elemento estructurante cuadrado de tamaño eeSize con todos los componentes iguales a 255 mediante una función de OpenCV provista para tal fin. Luego, se lleva a cabo una reconstrucción morfológica por dilatación geodésica mediante una función de skimage.
#@param img Imagen binaria en formato de arreglo de NumPy de dos dimensiones
#@param eeSize Tamaño del elemento estructurante que es utilizado para la eliminación de los residuos pequeños.
#@return Imagen binaria sin los residuos pequeños.
def eliminarResiduos(img, eeSize= 7):
    #primero eliminar pequenios residuos
    
    #version1: erosion + reconstruccion
    ee= np.ones((eeSize,eeSize))*255
    erosionada= cv.erode(img,ee)
    # reconstrucción por dilatacion geodésica para no perder nada
    limpia= np.uint8(morph.reconstruction(erosionada,img))
    
    #version2: apertura+dilatacion+and
#    iteraciones= 3
#    ee= np.ones((3,3))*255
#    limpia= cv.erode(img,ee,iterations=iteraciones)
#    cerrada= cv.dilate(limpia,ee,iterations=iteraciones+1)
#    #dilatacion para no perder nada
#    limpia= cv.bitwise_and(cerrada,img)
    return limpia


#%% Relleno agujeros
## Función que rellena los agujeros de una imagen binaria utilizando reconstrucción morfológica por dilatación geodésica.
#
# Se calcula el complemento de la imagen y se inicializa la semilla que serán utilizadas para la reconstrucción morfológica por dilatación geodésica mediante la función provista por skimage. Luego, se opera para obtener sólo los agujeros rellenos y se analizan los mismos utilizando la función de detección de bordes de cada componente conexa provista por OpenCV para saber si corresponde llenarlos o no.
#@param img Imagen binaria en formato de arreglo de NumPy de dos dimensiones.
#@param menoresA Tamaño máximo que puede tener un agujero interno a un cromosoma para que se rellene. Cuando es mayor, no se rellena. Si es 0, rellena todos los agujeros.
#@return Imagen binaria con los agujeros rellenos en formato de arreglo de NumPy de dos dimensiones.
def rellenoAgujeros(img, menoresA=0):
    #img= imagen binaria
    data= img.copy()
    #solo rellena por dilatacion geodesica a los agujeros de area menor a menoresA
    H,W= data.shape
    
    #complemento de la imagen
    Ic= 255-data
    #F= 1-I(x,y) si esta en el borde, sino 0. Modifico bordes
    F= np.zeros(data.shape, dtype= data.dtype)
    F[0,:]= 255-img[0,:]
    F[H-1,:]= 255-img[H-1,:]
    F[:,0]= 255-img[:,0]
    F[:,W-1]= 255-img[:,W-1]
    #complemento de la reconstruccion geodesica con F y Ic
    reconstruida= 255 - np.uint8(morph.reconstruction(F,Ic))
    
    if(menoresA==0): #no analiza nada
        return reconstruida
    else:  #analisis de agujeros!!
        #obtengo solo agujeros
        agujeros= reconstruida - data
        agujeros_chicos= np.zeros_like(agujeros)
        _, contours, _= cv.findContours(agujeros,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) #cv.CHAIN_APPROX_SIMPLE
        #para cada agujero
        for cnt in contours:
            areaAgujero= cv.contourArea(cnt)
            if(areaAgujero<menoresA):
                cv.drawContours(agujeros_chicos,[cnt],-1,255,cv.FILLED) 
        suma= cv.bitwise_or(img, agujeros_chicos)
        return suma
#    return np.uint8(morph.remove_small_holes(img,menoresA)*255)

#%% Analisis componentes conexas
## Función analiza cada componente conexa para determinar si corresponde a uno o más cromosomas, o a objetos no deseados.
#
# Primero se eliminan los objetos del borde con la función eliminarObjBorde(). Luego, utilizando la función de detección de bordes de cada componente conexa provista por OpenCV, se analiza el área de cada una de ellas. De ser necesario, se obtiene su convex hull con una función de OpenCV y se calcula la proporción entre ambas áreas para comparar si el objeto debe descartarse o no.
#@param img Imagen binaria en formato de arreglo de NumPy de dos dimensiones.
#@param umbralSegm Umbral que determina el tamaño máximo que puede tener el segmento en contacto con el borde de la imagen de un elemento. Si es mayor, se elimina.
#@param umbralArea Umbral que determina el tamaño máximo que puede tener un elemento para no ser considerado un posible residuo. Si es mayor, se utiliza el criterio del convex hull.
#@param umbralCH Umbral que determina la máxima proporción entre el área de un objeto y el área de su convex hull para no ser considerado un residuo. Si es mayor, se elimina.
#@return Lista que contiene los contornos de las componentes que no son residuos.
def compConexas(img, umbralSegm= 30, umbralArea= 4000, umbralCH= 0.8):
    #Sacar los bordes
    limpia= eliminarObjBorde(img,umbralSegm)
    
    #componentes conexas
    salida= [] #lista que contendra los contornos de las componentes que no son ruido
    _, contours, _= cv.findContours(limpia,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) #cv.CHAIN_APPROX_SIMPLE
    for cnt in contours:
        areaObj= cv.contourArea(cnt)
        #ver que cada uno no sea mayor al umbralArea
        if(areaObj>umbralArea):
#            circ= circularidad(cnt,areaObj)
            circ= areaObj/ cv.contourArea(cv.convexHull(cnt)) #areaObj / areaCH
            if(circ>umbralCH):
                #ruido
                continue
        salida.append(cnt)
    
    return salida


#%% objetos segmentados en vectores + normalizacion
## Función que se encarga de separar los objetos de una imagen en un vector de subimágenes.
#
# Para cada componente conexa de contours se calcula el mínimo rectángulo que la contiene con una función de OpenCV, se le aplica la máscara para la eliminación de otros objetos que pueda haber en el rectángulo, se le agrega un margen de dos píxeles a cada lado y luego se devuelven juntas en un vector de imágenes.
#@param data Imagen en escala de grises en formato de arreglo de NumPy de dos dimensiones.
#@param mask Máscara que indica los píxeles que pertecen al fondo y a los objetos.
#@param contours Lista en la que cada elemento es un vector que contiene los contornos de un objeto.
#@return Tupla de listas. En la primera, cada elemento es una imagen de un objeto segmentado, mientras que en la segunta está su correspondiente máscara que indica con 255 los píxeles donde está el cromosoma y con 0 donde es fondo.
def dividirROIs(data,mask,contours):
    salida= []
    salida_mask= []
    for cnt in contours:
        # bounding rect
        y,x,h,w = cv.boundingRect(cnt)
        margen= 2
        roi= mask[max(x-margen,0):x+w+margen+1,max(y-margen,0):y+h+margen+1]
        roi_mask= np.zeros_like(roi)
        offset=tuple(-np.min(cnt,axis=(0,1))+[margen,margen])
        roi_data= data[max(x-margen,0):x+w+margen+1,max(y-margen,0):y+h+margen+1] #+1 porque no cuenta el ultimo
        cv.drawContours(roi_mask,[cnt],-1,255,cv.FILLED,offset= offset) 
        # AND por si hay algun otro cromosoma cerca
#        roi_mask= cv.bitwise_and(roi_mask,roi)
        roi_data= cv.bitwise_and(roi_data,roi_mask)
        fondo= (255-roi_mask)
        # normalizacion
#        roi_data= cv.normalize(roi_data,roi_data,230,0,cv.NORM_MINMAX)
        roi_data= roi_data + fondo # para que sea fondo blanco
        roi_data= cv.normalize(roi_data,roi_data,255,0,cv.NORM_MINMAX)
        salida.append(roi_data)
        salida_mask.append(roi_mask)
    return (salida, salida_mask)


#%% sacar objetos del borde
## Función que elimina los objetos que están en contacto con el borde de una imagen verificando el tamaño del segmento que pertenece tanto al borde como al objeto.
#
# Se inicializa la imagen semilla que serán utilizadas para la reconstrucción morfológica por dilatación geodésica mediante la función provista por skimage. Antes de dicha reconstrucción, mediante erosiones con dos elementos estructurantes horizontal y vertical se eliminan de la semilla los elementos que tengan un segmento en contacto con el borde de la imagen menor al parámetro dado. Así, con la reconstrucción mencionada se obtienen los objetos a eliminar de la imagen binaria.
#@param img Imagen binaria.
#@param umbralSegm Umbral que determina el tamaño máximo que puede tener el segmento en contacto con el borde de la imagen de un elemento. Si es mayor, se elimina. Si es 0 elimina todos los objetos que están en el borde.
#@return Imagen binaria sin los objetos del borde que no cumplen el criterio mencionado.
def eliminarObjBorde(img,umbralSegm=0):
    #img binaria 0-255
    H,W= img.shape
    data= np.zeros_like(img)
    #inicializar bordes igual a img original
    data[0,:]= img[0,:]
    data[H-1,:]= img[H-1,:]
    data[:,0]= img[:,0]
    data[:,W-1]= img[:,W-1]
    #eliminar los segmentos menores a umbralSegm
    if(umbralSegm>0):
        vertEE= np.ones((1,umbralSegm+1))
        horizEE= np.ones((umbralSegm+1,1))
        soloVert= cv.erode(data,vertEE)
        soloHoriz= cv.erode(data,horizEE)
        suma= cv.bitwise_or(soloHoriz, soloVert)
        #sacar esquinas igual
        if(data[0,0]==255):
            suma[0,0]= 255
        if(data[H-1,0]==255):
            suma[H-1,0]= 255
        if(data[0,W-1]==255):
            suma[0,W-1]= 255
        if(data[H-1,W-1]==255):
            suma[H-1,W-1]= 255
        data= np.uint8(morph.reconstruction(suma,data))
    #reconstruccion geodesica a partir de eso
    objBorde= np.uint8(morph.reconstruction(data,img))
    #devolver sin objetos del borde
    return cv.bitwise_xor(img,objBorde)
