from matplotlib.pyplot import ginput
from matplotlib import pylab
from matplotlib import pyplot as plt
import pydicom
import numpy as np
import math
from funciones_p2 import *
from RegionGrowing import *

#SEMILLAS
def semillas(imagen, n):
    '''
    Semillas de la imagen.
    
    Parámetros:
        imagen : matriz de la imagen.
        n: número de semillas.
        
    Devuelve:
        coord_semilla : tupla que contiene las coordenadas de la semilla.
    
    Notas:
        Se crea una lista vacía donde se almacenarán las coordenadas.
        Se muestra la imagen y se pincha en la zona de interés. Las coordenadas de esta zona se obtienen
        mediante ginput y se almacenan en la variable semilla. A continuación se redondean los valores para
        obtener coordenadas de valor entero. 
        Por último se colocan en el orden adecuado,es decir, (x1,x2) ya que ginput devuelve el formato (x2, x1)
        y se añaden a la lista creada.
    '''
    semillas_list = [] #Lista vacía.
    
    plt.figure() #Crea una nueva figura.
    plt.imshow(imagen, cmap=pylab.cm.gray) #Muestra los datos como una imagen.
    plt.show() #Muestra la figura.
    semilla = ginput(n) #Se hace clic n veces en la figura y devuelve las coordenadas de cada clic en una lista.
    semilla = np.array(semilla) #Convierte la lista en array.
    
    #Se recorre el array semilla.
    for coord in semilla:
        x1 = math.ceil(coord[0]) #Redondea la coordenada x2.
        x2 = math.ceil(coord[1]) #Redondea la coordenada x1.
   
        coord_semilla = (x2,x1) #Se colocan las coordenadas en el orden correcto (x1, x2) no (x2, x1).
        semillas_list.append(coord_semilla) #Se añade la coordenada a la lista.
        
    return semillas_list 


#------------------------------------EXTRACCIÓN DE SEMILLAS-----------------------------------------
# img_norm1 = normalizar('imagen1.dcm') #Se normaliza la imagen.
# list_sem1 = semillas(img_norm1, 8)

#img_norm2 = normalizar('imagen2.dcm') #Se normaliza la imagen.
#list_sem2 = semillas(img_norm2, 1)

#img_norm3 = normalizar('imagen4.dcm') #Se normaliza la imagen.
#list_sem3 = semillas(img_norm3, 15)