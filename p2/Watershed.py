from matplotlib import pylab
from matplotlib import pyplot as plt
import pydicom
import numpy as np
import random
import skimage
from skimage import filters
from skimage.segmentation import watershed
from funciones_p2 import *
from imimposemin import *

def WatershedExerciseP2(imagen_normalizada, imagen_sobel, coord_sem): 
    '''
    Adición de ruido gaussiano.
    
    Parámetros:
        imagen_normalizada : matriz de píxeles de la imagen normalizada.
        imagen_sobel : matriz de pixeles de la imagen con filtro de sobel.
        coord_sem : lista con las coordenadas de las semillas.
        
    Devuelve:
        img_ws1 : matriz de píxeles resultado de la aplicación de Watershed. A la imagen se le ha aplicado la función imimposemin. 
        img_ws2 : matriz de píxeles resultado de la aplicación de Watershed. La imagen es aquella que presenta filtro de sobel.
    
    Notas:
        Se crea la matriz de ceros del tamaño de la imagen original. A continuación se pincha en la imagen, se calculan 
        las coordenadas de las semillas y se llevan a blanco los pixeles que se encuentran en las coordenas de las semillas pero,
        en este caso, en la matriz de ceros creada.
        Se aplican los algortimos imimposemin (sobre la imagen de sobel y la binaria) y watershed (sobre la imagen de imimposemin
        y sobel). Por último, se muestra en pantalla el resultado de la función imimposemin sobre la imagen normalizada y la binaria. 
    '''

    img_binaria = np.zeros(imagen_normalizada.shape) #Matriz de ceros del tamaño de la imagen de entrada.

    for elem in range (len(coord_sem)):
        coord = coord_sem[elem] #Coordenada i de la lista
        fila = coord[0] #Fila
        col = coord[1] #Columna
        img_binaria[fila][col] = 1 #Se establece como 1 la coordenada de la semilla.
        
    min_loc = imimposemin(imagen_sobel, img_binaria) #Mínimos locales 
    img_ws1 = watershed(min_loc) #Watershed con imimposemin
    img_ws2 = watershed(imagen_sobel) #Watershed sin imimposemin
    
    min_norm = imimposemin(imagen_normalizada, img_binaria) #Mínimos locales 
    plt.figure(figsize=[8,6]) 
    plt.imshow(min_norm,cmap=pylab.cm.gray) #Alpha para visualización de la máscara generada sobre la imagen original.
    plt.title('IMIMPOSEMIN en NORMALIZADA')

    return img_ws1, img_ws2