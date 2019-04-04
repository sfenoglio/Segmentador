#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package interfaz
#  Paquete que incluye las funciones y definiciones necesarias para manejar la interfaz de usuario de la herramienta.
# Los parámetros que acepta la interfaz se definen mediante la librería argparse y se utiliza JSON para almacenar los parámetros por defecto.
# Luego, se definen funciones que se encargan del cargado y almacenamiento de las imágenes de entrada y salida.

"""
Created on Tue Mar  5 23:22:48 2019

@author: sebastian
"""

import argparse
import json
import os
import numpy as np
import cv2 as cv

#%% Inicializar parser y leer parametros por defecto
parser= argparse.ArgumentParser()
try:
    with open('./config/default_args.json') as json_data:
      default_args = json.load(json_data)
except:
    print("No se pueden leer los parámetros por defecto.")
    quit()


#%% -------------------------------------------------------------------------------------------------------------------
#                                                   Parametros de interfaz
#   -------------------------------------------------------------------------------------------------------------------
# path a la imagen
parser.add_argument("--path_img", 
                    help= "Path de la imagen que se quiere segmentar. Por defecto, se toma una imagen del dataset explicado en el informe.",
                    default= default_args['path_img'])

#path_out
parser.add_argument("--path_out", 
                    help="Path de salida. Si no existe la carpeta, se crea. Por defecto, se crea la carpeta 'salida' en el directorio que se encuentre el segmentador.",
                    default= default_args['path_out'])

# Default args
parser.add_argument("--saveDefaultArgs", 
                    help="Parámetro que indica si se quiere actualizar los parámetros por defecto de la herramienta. Por defecto no se aplica.",
                    action='store_true')

# Ver params por defecto
parser.add_argument("--verDefaultArgs", 
                    help="Muestra los parámetros por defecto de la herramienta sin ejecutarla. Por defecto no se aplica.",
                    action='store_true')

# restoreDefaultArgs
parser.add_argument("--restoreDefaultArgs", 
                    help="Restaura los valores por defecto sin ejecutar la herramienta. Por defecto no se aplica.",
                    action='store_true')


#%% -------------------------------------------------------------------------------------------------------------------
#                                                 Parametros de pre-procesamiento 
#   -------------------------------------------------------------------------------------------------------------------
#tamCuadFondo
parser.add_argument("--tamCuadFondo", 
                    help="Tamaño del cuadrado del fondo que se toma de las esquinas en el pre-procesamiento.",
                    type= int,
                    default= default_args['tamCuadFondo'])

#tamCuadUmbral
parser.add_argument("--tamCuadUmbral",
                    help= "Lista que contiene los tamaños de la ventana cuadrada que se utiliza en el umbral adaptado en el pre-procesamiento. Si es 0, calcula el umbral de Otsu sobre toda la imagen.",
                    type= list,
                    default= default_args['tamCuadUmbral'])

#maxTamAgujero
parser.add_argument("--maxTamAgujero",
                    help= "Tamaño máximo que puede tener un agujero interno a un cromosoma para que se rellene su máscara en el pre-procesamiento. Cuando es mayor, no se rellena.",
                    type= int,
                    default= default_args['maxTamAgujero'])

#eeSize
parser.add_argument("--eeSize",
                    help= "Tamaño del elemento estructurante que es utilizado para la eliminación de los residuos pequeños en el pre-procesamiento.",
                    type= int,
                    default= default_args['eeSize'])

#umbralSegm
parser.add_argument("--umbralSegm",
                    help= "Umbral que determina el tamaño máximo que puede tener el segmento en contacto con el borde de la imagen de un elemento en el pre-procesamiento. Si es mayor, se elimina. Si es 0 elimina todos los objetos que están en el borde.",
                    type= int,
                    default= default_args['umbralSegm'])

#umbralArea
parser.add_argument("--umbralArea",
                    help= "Umbral que determina el tamaño máximo que puede tener un elemento para no ser considerado un posible residuo en el pre-procesamiento. Si es mayor, se utiliza el criterio del convex hull.",
                    type= int,
                    default= default_args['umbralArea'])

#umbralCH
parser.add_argument("--umbralCH",
                    help= "Umbral que determina la máxima proporción entre el área de un objeto y el área de su convex hull para no ser considerado un residuo en el pre-procesamiento. Si es mayor, se elimina.",
                    type= float,
                    default= default_args['umbralCH'])

#TilesGridSize
parser.add_argument("--TilesGridSize",
                    help= "Tamaño de las ventanas cuadradas que se utilizar para aplicar CLAHE (Contrast-Limited Adaptive Histogram Equalization) en el pre-procesamiento.",
                    type= int,
                    default= default_args['TilesGridSize'])

#ClipLimit
parser.add_argument("--ClipLimit",
                    help= "Límite para el contraste utilizado para aplicar CLAHE (Contrast-Limited Adaptive Histogram Equalization) en el pre-procesamiento.",
                    type= int,
                    default= default_args['ClipLimit'])

#guardarClusters
parser.add_argument("--guardarPreproc",
                    help= "Indica que se guarde el cluster detectado en el pre-procesamiento. Por defecto se guarda.",
                    action= 'store_true')
parser.add_argument("--noGuardarPreproc",
                    help= "Indica que NO se guarde el cluster detectado en el pre-procesamiento. Por defecto se guarda.",
                    action= 'store_true')


#%% -------------------------------------------------------------------------------------------------------------------
#                                                 Parametros de separacion 
#   -------------------------------------------------------------------------------------------------------------------
#path_model
parser.add_argument("--path_model",
                    help= "Path del archivo que contiene los parámetros entrenados correspondiente a la arquitectura path_arch",
                    default= default_args['path_model'])

#path_arch
parser.add_argument("--path_arch",
                    help= "Path del modelo de la red convolucional utilizada para la separación de solapamientos",
                    default= default_args['path_arch'])

#device
parser.add_argument("--device",
                    help= "Indica que la predicción se haga mediante CPU. Por defecto, se realiza por GPU.",
                    choices= {"cpu","gpu"},
                    default= default_args['device'])

#adecStd
parser.add_argument("--adecStd",
                    help= "Indica que la predicción se haga mediante CPU. Por defecto, se realiza por GPU.",
                    type=int,
                    default= default_args['adecStd'])

#adecBlur
parser.add_argument("--adecBlur",
                    help= "Indica que la predicción se haga mediante CPU. Por defecto, se realiza por GPU.",
                    type= float,
                    default= default_args['adecBlur'])

#tamImagen
parser.add_argument("--tamImagenSeparar",
                    help= "Indica el tamaño cuadrado al que se amplía la imagen de un cluster antes de pasar por la red convolucional. Por defecto, es 256.",
                    type= int,
                    default= default_args['tamImagenSeparar'])

#noAplicarMaskPreproc 
parser.add_argument("--aplicarMaskPreproc", #-> es el que se usa para guardar los valores por defecto
                    help= "Indica que se aplique la máscara de pre-procesamiento a la predicción de la red convolucional. Por defecto, se aplica.",
                    action= 'store_true')
parser.add_argument("--noAplicarMaskPreproc",
                    help= "Indica que NO se aplique la máscara de pre-procesamiento a la predicción de la red convolucional. Por defecto, se aplica.",
                    action= 'store_true')

#invertirImg 
parser.add_argument("--invertirImg", #-> es el que se usa para guardar los valores por defecto
                    help= "Indica que se calcule la inversa de la imagen antes del pasaje a la red convolucional. Por defecto, se aplica.",
                    action= 'store_true')
parser.add_argument("--noInvertirImg",
                    help= "Indica que NO se calcule la inversa de la imagen antes del pasaje a la red convolucional. Por defecto, se aplica.",
                    action= 'store_true')


#%% -------------------------------------------------------------------------------------------------------------------
#                                              Parametros de post-procesamiento 
#   -------------------------------------------------------------------------------------------------------------------
#umbralCC
parser.add_argument("--umbralCC", 
                    help="Umbral utilizado para la determinación de 'clases confiables' para la corrección por canales aplicada en el post-procesamiento. Si es 0 no se hace.",
                    type= float,
                    default= default_args['umbralCC'])

#umbralKNN
parser.add_argument("--umbralKNN", 
                    help="Umbral utilizado para la determinación de 'píxeles confiables' para la corrección mediante el algoritmo de k-nn usado en el post-procesamiento. Por defecto es 0 y no se hace.",
                    type= float,
                    default= default_args['umbralKNN'])

#noAplicarEPI
parser.add_argument("--umbralEPI", 
                    help="Umbral utlizado realizar la corrección de pequeñas imperfecciones del post-procesamiento. Por defecto es 100. Si es 0, no se aplica EPI.",
                    type= int,
                    default= default_args['umbralEPI'])

args= parser.parse_args()


#%% -------------------------------------------------------------------------------------------------------------------
#                                               Parametros que requieren control
#   -------------------------------------------------------------------------------------------------------------------
#Mostrar argumentos
if(args.verDefaultArgs):
    print("Parámetros por defecto:")
    import pprint
    pprint.pprint(default_args)
    #print(json.dumps(default_args,indent=4,sort_keys=True))    
    quit()

#restaurar por defecto
if(args.restoreDefaultArgs):
    try:
        with open('./config/original_default_args.json') as json_data:
            original_default_args = json.load(json_data)
    except:
        print("No se pueden leer los parámetros por defecto originales.")
        quit()
    
    with open('./config/default_args.json', 'w') as outfile:
        json.dump(original_default_args, outfile)
    print("Argumentos por defecto originales restaurados.")
    quit()

#control de los opuestos
if(args.aplicarMaskPreproc and args.noAplicarMaskPreproc): #los dos no pueden ser verdadero
    print("Error con los argumentos aplicarMaskPreproc y noAplicarMaskPreproc: NO pueden aplicarse en simultáneo.")
    quit()
elif(not args.aplicarMaskPreproc or args.noAplicarMaskPreproc): #uno de los dos es falso, si son los dos: cargo el por defecto
    args.aplicarMaskPreproc= default_args['aplicarMaskPreproc']
# si aplicarMaskPreproc fuese veradero ya tiene el valor, y si noAplicarMaskPreproc lo fuese: aplicarMaskPreproc ya es falso

if(args.invertirImg and args.noInvertirImg): #los dos no pueden ser verdadero
    print("Error con los argumentos invertirImg y noInvertirImg: NO pueden aplicarse en simultáneo.")
    quit()
elif(not args.invertirImg or args.noInvertirImg): #uno de los dos es falso, si son los dos: cargo el por defecto
    args.invertirImg= default_args['invertirImg']
# si invertirImg fuese veradero ya tiene el valor, y si noInvertirImg lo fuese: invertirImg ya es falso

if(args.guardarPreproc and args.noGuardarPreproc): #los dos no pueden ser verdadero
    print("Error con los argumentos guardarPreproc y noGuardarPreproc: NO pueden aplicarse en simultáneo.")
    quit()
elif(not args.guardarPreproc or args.noGuardarPreproc): #uno de los dos es falso, si son los dos: cargo el por defecto
    args.guardarPreproc= default_args['guardarPreproc']
# si guardarPreproc fuese veradero ya tiene el valor, y si noGuardarPreproc lo fuese: guardarPreproc ya es falso
        

#%% -------------------------------------------------------------------------------------------------------------------
#                                                       Importacion de modulos
#   -------------------------------------------------------------------------------------------------------------------
#importar modelo
try:
    exec("from src import "+args.path_arch)
except:
    print("La arquitectura indicada no existe.")
    quit()
#inicializar modelo
try:
    exec("model= "+ args.path_arch + "." + args.path_arch + "()")
except:
    print("La arquitectura indicada no puede inicializarse. Verificar que el nombre de la clase que define la arquitectura sea el mismo que el del archivo")
    quit()

#verificar existencia de archivo
if(not os.path.exists(args.path_img)):
    print(args.path_img)
    print("El archivo indicado no existe.")
    quit()
    
#preprocesamiento
try:
    from src import preprocesamiento as pre
except:
    print("No se encuentra módulo de pre-procesamiento o una librería asociada a él.")
    quit()

#separación
try:
    from src import separacion
except:
    print("No se encuentra módulo de separación o una librería asociada a él.")
    quit()
if(args.device=="cpu"):
    sep= separacion.separacion(args.path_model, model, gpu= False)
else:
    sep= separacion.separacion(args.path_model, model, gpu= True)
del model

#postprocesamiento
try:
    from src import postprocesamiento as post
except:
    print("No se encuentra módulo de post-procesamiento o una librería asociada a él.")
    quit()


#%%
## Función que guarda los parámetros utilizados en la ejecución como parámetros por defecto.
#
# Éstos son almacenados con formato JSON en el archivo 'default_args.json' ubicado en la carpeta 'config' de la herramienta.
def guardarDefaultArgs():
    argparse_dict = vars(args)
    with open('./config/default_args.json', 'w') as outfile:
        json.dump(argparse_dict, outfile) 


#%%
## Función que se encarga de leer del disco las imágenes de entrada a la herramienta.
#
# Acepta imágenes del tipo [".jpg", ".bmp", ".png", ".tiff", ".jpeg", ".tif"] o un archivo ".txt" en el que haya en cada línea un path a las imágenes 
#con el formato mencionado anteriormente. Esto último es para permitir procesar muchas imágenes a la vez.
#@param path Directorio del archivo a procesar. Se verifica que al menos tenga 7 caracteres.
#@param path_out Directorio de salida, utilizado para generar un path de salida para cada imagen de entrada ya que se crea una carpeta para cada una de ellas.
#@return Tupla de dos listas. La primera con las imágenes cargadas y la segunda con el path de salida para cada una de ellas.
def leerImagenes(path, path_out):
    #el largo minimo debe ser 7: una letra mas extension + /n
    if(len(path)<7):
        print("Insertar path válido.")
        quit()
    
    #tipos soportados por OpenCV
    tipos= [".jpg", ".bmp", ".png", ".tiff", ".jpeg", ".tif"]
    
    #si es carpeta/archivo.txt, leer todas las que sean .tiff, .png, .jpg, .jpeg
    lista= []
    if(path[-4:]==".txt"):
        #obtener lista. Abre archivo (y cierra cuando termine lectura)
        with open(path) as fichero:
            # recorre línea a línea el archivo
            for linea in fichero:
                # guardar línea sacando "/n"
                lista.append(linea[:-1])
    else:
        #sino cargar esa sola imagen
        lista.append(path)
    
    #verificar tipos    
    imgs= []
    paths_out= []
    for l in lista:
        coincidio= False
        for tipo in tipos:
            if(l[-(len(tipo)):]==tipo):
                #si coincide devuelvo tupla de listas (imagen, path_out). Se verifica antes si existe
                imgs.append(cv.imread(l,0))
                
                #buscar "/"
                desde= 0
                for k in range(len(l)-len(tipo),-1,-1):
                    if(l[k]=="/"):
                        desde= k+1
                        break
                        
                paths_out.append(path_out+l[desde:-(len(tipo))]+"/") # "/" para que sea carpeta
                coincidio= True
                break
            
        #si no coincide, avisar
        if(not coincidio):
            print("Formato de imagen no válido en: "+l)
    
    #verificar que haya algo
    if(len(imgs)==0):
        print("No hay imágenes válidas.")
        quit()
        
    #devolver lista de imagenes + lista de paths de salida
    return (imgs, paths_out)

#%%
## Función que se encarga de almacenar las imágenes de salida que genera la herramienta.
#
# Para cada imagen, se busca en su correspondiente máscara las clases que ésta contiene.
# Para cada clase, se genera una máscara y se aplica a la imagen para obtener sólo el cromosoma de esa clase.
# Esta última imagen es almacenada en la carpeta de salida en formato '.png' con un nombre del tipo 'nn_cc_kk.png', donde
# nn es el número de cluster, cc el número de clase y kk un entero incremental para evitar repeticiones.
#@param imgs Lista con las imágenes a guardar en formato de arreglo de NumPy.
#@param mascaras Lista con las máscaras correspondiente a cada imagen de imgs en formato de arreglo de NumPy.
#@param path_out String que es el directorio de salida de las imágenes de salida.
def guardarImagenes(imgs, mascaras, path_out):
    #para cada roi
    #clases= [1 for i in range(0,24)]
    for i in range(len(imgs)):
        mascara= mascaras[i]
        img= imgs[i]
        #guardar: hay que sacar las máscaras de cada clase
        valores, cantidades= np.unique(mascara[mascara>0], return_counts=True)
        #para cada valor
        for v in valores:
            #para no guardar el fondo
            if(v!=0):
                #creo mascara y aplico
                mask= np.zeros_like(mascara, dtype= np.uint8)
                mask[mascara==v] = 255
                parcial= cv.bitwise_and(mask,img)
                
                #recortar
                #rectangulo alrededor
                y,x,h,w= cv.boundingRect(cv.findNonZero(parcial))
                margen= 5
                salida= parcial[max(0,x-margen):min(x+w+margen,parcial.shape[0]),max(0,y-margen):min(y+h+margen,parcial.shape[1])].copy()
                #guardar
                #cv.imwrite(path_out+str(i).zfill(2)+"_"+str(v).zfill(2)+"_"+str(clases[v]).zfill(2)+".png",
                #           255-salida)
                cv.imwrite(path_out+str(i).zfill(2)+"_"+str(v).zfill(2)+".png",
                           255-salida)
                #clases[v]+=1 #para no repetirlas
