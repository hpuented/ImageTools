from matplotlib.pyplot import ginput
from matplotlib import pylab
from matplotlib import pyplot as plt
import pydicom
import numpy as np
import math
from funciones_p2 import *


#SEGMENTACIÓN POR CRECIMIENTO DE REGIONES
def RegionGrowingP2(img_norm, list_sem, umbral):
    '''
    Segmentación por crecimiento de regiones.
    
    Parámetros:
        imagen : matriz de la imagen normalizada.
        list_sem : lista de coordenadas de las semillas.
        umbral: valor determinante del rango de grises incluidos en el criterio de homogeneidad. 
        
    Devuelve:
        matriz_cr : matriz de la imagen segmentada.
    
    Notas:
        Se normaliza la imagen y se obtienen las coordenadas de la semilla. A continuación se crea
        una matriz de ceros del tamaño de la imagen normalizada y se establece con valor 1 la coordenada 
        donde se ha clicado. Se analizan los 8 vecinos iterativamente hasta que los píxeles no cumplan
        la condición.
    '''
    
    coord_sem = list_sem[0] #Coordenada de la semilla.
    matriz_cr = np.zeros(img_norm.shape) #Matriz de ceros del tamaño de la imagen normalizada.
    matriz_cr[coord_sem[0]][coord_sem[1]] = 1 #Se establece como 1 la coordenada de la semilla.
    pix_sem = img_norm[coord_sem[0]][coord_sem[1]] #Nivel de gris de la semilla.
    
    coord_homog = [] #Lista vacía.
    coord_homog.append(coord_sem) #Se añaden a la lista las coordenadas de la semilla.
    
    ya_comparadas = [] #Lista vacía.
    ya_comparadas.append(coord_sem) #Se añaden a la lista las coordenadas de la semilla.
    
    #Ejecución de instrucciones mientras se ejecuta la condición. 
    while (len(coord_homog)!= 0):
        coordenada = coord_homog.pop(0) #Coordenada de la lista que se ha eliminado, es el píxel a centrar.
        for i in range(coordenada[0]-1, coordenada[0]+2): #Filas.
            for j in range (coordenada[1]-1, coordenada[1]+2): #Columnas.
            #Se cogen coordenadas con vecindad a 8 centradas en la semilla.
                if ((0 <= i <= (img_norm.shape[0]-1)) and (0 <= j <= (img_norm.shape[1]-1))): 
                    pixel = img_norm[i][j] #Valor de gris de cada pixel para comparar.
                    c = (i,j) #Coordenada del píxel vecino.
                
                    if c not in ya_comparadas:
                        ya_comparadas.append(c) #Se añade la coordenada a la lista.
                        if (pixel >= (pix_sem - umbral)) and (pixel <= (pix_sem + umbral)):
                        #En este if hay 3 condiciones:
                            #1. Si el píxel pertenece al intervalo.
                            #2. Si el píxel no está fuera de los límites de la imagen.
                            #3. Si la coordenada del vecino NO está aún en ya_comparadas.
                        
                            matriz_cr[i][j] = 1 #Se manda a blanco el valor del píxel.
                            coord_homog.append(c) #Se añade la coordenada a la lista.
                        
    return matriz_cr