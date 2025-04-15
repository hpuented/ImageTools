from matplotlib.pyplot import ginput
from matplotlib import pylab
from matplotlib import pyplot as plt
import pydicom
import numpy as np
import math
import skimage
from skimage import filters
from skimage.util import pad


#NORMALIZACIÓN.
def normalizar(imagen):
    '''
    Normalización de la imagen.
    
    Parámetros:
        imagen : nombre del archivo (.dcm)
        
    Devuelve:
        matriz_norm : matriz de píxeles de la imagen normalizada (valores entre 0 y 1).
    
    Notas:
        Se lee el archivo y se extrae la matriz de píxeles. Si hay algún píxel con valor negativo, 
        se le suma el mínimo valor de la imagen, haciendo así que toda la matriz tenga valores positivos.
        Por último, para que dichas intensidades de píxel estén comprendidas entre 0 y 1, se divide 
        cada píxel entre el máximo valor de la matriz.
    '''
    
    dataset = pydicom.dcmread(imagen)
    matriz_pixel = dataset.pixel_array #Matriz de píxeles.
    valor_min = np.amin(matriz_pixel) #Valor mínimo de píxel en esta matriz.

    #Normalización en 2 pasos.
    if valor_min < 0: #Si la matriz tiene algún valor negativo. 
        
        #PAS0 1: Sumamos el valor mínimo a cada elemento de la matriz.
        matriz_valormin = -(valor_min)*np.ones(matriz_pixel.shape)
        matriz_suma = matriz_pixel + matriz_valormin #Pixels de la matriz positivos.

        #PASO 2: Dividimos cada píxel de la matriz entre su máximo valor.
        valor_max = np.amax(matriz_suma) 
        matriz_norm = (1/valor_max)*matriz_suma #Valores de píxel entre 0 y 1.

    else: 
        #Dividimos cada píxel de la matriz entre su máximo valor.
        valor_max = np.amax(matriz_pixel)
        matriz_norm = (1/valor_max)*matriz_pixel #Valores de píxel entre 0 y 1.

    return matriz_norm



#FILTRO ANISOTRÓPICO
def anisotropico(imagen, umbral, iteraciones):
    '''
    Filtro anisotrópico.
    
    Parámetros:
        imagen : matriz de píxeles normalizada con ruido. 
        umbral : parámetro a comparar con el gradiente. 
        iteraciones : número de veces que pasa la imagen por el filtro. 
        
    Devuelve:
        img_anisotropico : matriz de píxeles de la imagen filtrada con el filtro anisotrópico. 
    
    Notas:
        Se coge una imagen con ruido y se calcula el gradiente de cada píxel, con una máscara de Sobel. 
        A continuación, se compara cada gradiente con un umbral deseado y en función del valor, 
        se pasa el píxel a través de un filtro de media o se mantiene. El proceso se realiza varias veces. 
    '''
    
    contador = 0 #Contador de iteraciones.
    img_sobel = skimage.filters.sobel(imagen) #Filtro derivativo (Máscara Sobel).
    img_padding = pad(imagen, (1,1), 'edge') #Imagen ruidosa con padding. 

    while contador < iteraciones: 
        if contador == 0: #Coge la imagen con ruido.
            
            img_anisotropico = np.ones(imagen.shape) #Matriz de 1. 
        
            for i in range(img_sobel.shape[0]): #Itera la matriz de la imagen de Sobel - Filas.
                for j in range(img_sobel.shape[1]): #Itera la matriz de la imagen de Sobel - Columnas.
        
                    if img_sobel[i][j] > umbral: #Gradiente de cada pixel ruidoso (se encuentra en la img_sobel).
                        img_anisotropico[i][j] = imagen[i][j]  #Pixel de la imagen con ruido con borde.
            
                    else: 
                        parche = img_padding[i:i+3, j:j+3] #Parche 3x3.
                        img_anisotropico[i][j] = np.mean(parche) #Media del parche.
        
            contador = contador + 1  
        
        else: #Coge la imagen filtrada y se repite el mismo proceso. 
            img_sobel = skimage.filters.sobel(img_anisotropico)
            img_padding = pad(img_anisotropico, (1,1), 'edge')
       
            for i in range(img_sobel.shape[0]): 
                for j in range(img_sobel.shape[1]): 
        
                    if img_sobel[i][j] > umbral: 
                        img_anisotropico[i][j] = img_anisotropico[i][j]
            
                    else:
                        parche = img_padding[i:i+3, j:j+3] 
                        img_anisotropico[i][j] = np.mean(parche) 
    
            contador = contador + 1 #Seguimos con la siguiente iteración. 
            
    return img_anisotropico