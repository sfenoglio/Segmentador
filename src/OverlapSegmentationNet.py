#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package Red W
#  Paquete que incluye la clase 'RedW' que define la arquitectura de la red similiar a la U.
#
# Se hereda de la clase 'nn.Module' de PyTorch y se sobrecarga la función que se encarga de hacer el pasaje hacia adelante de la imagen 'forward()'.

"""
Created on Tue Sep 18 14:37:54 2018

@author: sebastian
"""

import torch
from torch import nn
import resource


class OverlapSegmentationNet(nn.Module):
    #%%
    ## Constructor que define la estructura de la red convolucional.
    def __init__(self, canalesEntrada= 1):
        print("Cargando modelo...")

        #inicializar con clase de la que hereda
        super(OverlapSegmentationNet, self).__init__()
        self.canalesEntrada= canalesEntrada
        
        inplace= True
        padding = 1
        #Bloque 1
        self.layer1 = nn.Sequential(
                nn.Conv2d(self.canalesEntrada, 64, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(64),
                nn.ReLU(inplace))
        self.layer2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Bloque 2
        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace))
        self.layer4= nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Bloque 3
        self.layer5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(256),
                nn.ReLU(inplace),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(256),
                nn.ReLU(inplace))
        self.layer6= nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Bloque 4
        self.layer7 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(512),
                nn.ReLU(inplace),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(512),
                nn.ReLU(inplace))
        self.layer8= nn.MaxPool2d(kernel_size=2, stride=2)
        
#        #Bloque 5
        self.layer9 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(512),
                nn.ReLU(inplace),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(512),
                nn.ReLU(inplace))
        


        #Deconv Bloque 1
        self.layer10= nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.layer11= nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(512),
                nn.ReLU(inplace),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace))

        #Deconv Bloque 2
        self.layer12= nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.layer13= nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(256),
                nn.ReLU(inplace),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace))

        #Deconv Bloque 3
        self.layer14= nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.layer15= nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace))
        
        #Deconv Bloque 4
        self.layer16= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.layer17= nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(64),
                nn.ReLU(inplace),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=padding), 
                nn.BatchNorm2d(64),
                nn.ReLU(inplace))
        
        #%% Output convolution. Number of filters should equal number of channels of the output
        self.layer18= nn.Conv2d(64, 24, kernel_size=1, stride=1, padding=0)


            
        print("Modelo cargado. (Memoria: "+str((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024)+"MB)")
            
    
    
    #%% pasada hacia adelante.
    ## Sobrecarga de la función homóloga de PyTorch que realiza la pasada hacia adelante por la red convolucional. 
    #
    # Se encarga de determinar cómo se interrelacionan entre sí las capas definidas anteriormente en el constructor.
    #@param x Entradas en formato Tensor de 4 dimensiones de PyTorch. La primera corresponde a la cantidad de datos, la segunda a la cantidad de canales y las últimas dos al tamaño de la imagen.
    #@return Salida de la red convolucional en formato Tensor de 4 dimensiones de PyTorch. La primera corresponde a la cantidad de datos, la segunda a la cantidad de canales (24 por la cantidad de clases) y las últimas dos al tamaño de la imagen.
    def forward(self, x):
        out1= self.layer1(x)
        out = self.layer2(out1)
        out2= self.layer3(out)
        out = self.layer4(out2)
        out3= self.layer5(out)
        out = self.layer6(out3)
        #out= self.layer7(out)
        out4= self.layer7(out)
        out = self.layer8(out4)
        out = self.layer9(out)

        out = self.layer10(out)
        out = torch.cat((out4, out), 1) #axis= 2
        del out4
        out = self.layer11(out)

        out = self.layer12(out)
        out = torch.cat((out3, out), 1) #axis= 2
        del out3
        out = self.layer13(out)

        out = self.layer14(out)
        out = torch.cat((out2, out), 1) #axis= 2
        del out2
        out = self.layer15(out)

        out = self.layer16(out)
        out = torch.cat((out1, out), 1) #axis= 2
        del out1
        out = self.layer17(out)
        
        return self.layer18(out)
