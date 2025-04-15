from matplotlib import pylab
from matplotlib import pyplot as plt
import pydicom
import numpy as np
import random
import skimage
from skimage.transform import resize
from sklearn.feature_extraction import image
from skimage.util import pad
from skimage.restoration import denoise_nl_means
from skimage import data
from skimage import filters
from numba import prange, njit #Se runea en multiples CPUs


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
        matriz_valormin = -(valor_min)*np.ones((512, 512))
        matriz_suma = matriz_pixel + matriz_valormin #Pixels de la matriz positivos.

        #PASO 2: Dividimos cada píxel de la matriz entre su máximo valor.
        valor_max = np.amax(matriz_suma) 
        matriz_norm = (1/valor_max)*matriz_suma #Valores de píxel entre 0 y 1.

    else: 
        #Dividimos cada píxel de la matriz entre su máximo valor.
        valor_max = np.amax(matriz_pixel)
        matriz_norm = (1/valor_max)*matriz_pixel #Valores de píxel entre 0 y 1.

    return matriz_norm


#ADICIÓN DE RUIDO IMPULSIVO (SAL PIMIENTA).
def ruido_SP(imagen, porcentaje_ruido): #El valor de porcentaje_ruido añade más o menos ruido a la imagen
    '''
    Adición de ruido impulsivo (sal pimienta).
    
    Parámetros:
        imagen : matriz de píxeles normalizada.
        porcentaje_ruido : valor que determina la cantidad de ruido que tendrá la imagen.
        
    Devuelve:
        img_SP : matriz de píxeles de la imagen con ruido impulsivo
    
    Notas:
        Se crea una matriz vacía con la misma dimensión que la imagen normalizada (es importante que la 
        matriz esté normalizada para que los niveles de grises vayan de 0, negro absoluto, a 1, blanco
        absoluto). Se recorre dicha imagen y en función de un valor aleatorio se reemplazará el valor del 
        píxel por 0, 1 o se mantendrá intensidad original. 
    '''

    img_SP = np.ones(imagen.shape) #Creamos matriz de 1. 
    umbral = 1-porcentaje_ruido #Umbral relacionado con porcentaje de ruido. 

    #Se recorre la imagen. 
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            rdn = random.random() #Valores aleatorios entre 0, 1.
            
            if rdn < porcentaje_ruido:
                img_SP[i][j] = 0 #Se cambia por negro absoluto.

            elif rdn > umbral:
                img_SP[i][j] = 1 #Se cambia por blanco absoluto. 

            else:
                img_SP[i][j] = imagen[i][j] #Se mantiene la intensidad de píxel. 

    return img_SP


#ADICIÓN DE RUIDO GAUSSIANO.
def ruido_gaussiano(imagen, media, sigma): #Cuanto menor sea el parámetro sigma, menos ruido se añade a la imagen
    '''
    Adición de ruido gaussiano.
    
    Parámetros:
        imagen : matriz de píxeles normalizada.
        media : media de la gaussiana.
        sigma : desviación típica de la gaussiana. 
        
    Devuelve:
        img_gauss : matriz de píxeles de la imagen con ruido gaussiano. 
    
    Notas:
        Se crea la matriz de gauss que muestra la densidad de probabilidad que responde a una 
        distribución normal (gaussiana) cuyos parámetros son los argumentos de la función (media,
        sigma). Por último, se le suma a la matriz normalizada.  
    '''

    dimension = imagen.shape #Dimensión de la matriz.
    gauss = np.random.normal(media,sigma,dimension) #Distribución normal.
    img_gauss = imagen + gauss #Imagen con ruido. 

    return img_gauss


#REDIMENSIÓN DE LA IMAGEN. 
def redimensionar(imagen, dimension):
    '''
    Redimensionado de la imagen. 
    
    Parámetros:
        imagen : matriz de píxeles normalizada con o sin ruido en función de lo que se necesite. 
        dimensión :  nueva dimensión que va a tener la imagen.
        
    Devuelve:
        img_redim :  matriz de píxeles redimensionada. 
    
    Notas:
        Usamos directamente una función.         
    '''
    
    img_redim = resize(imagen, (dimension, dimension))

    return img_redim


#NON LOCAL MEANS.
@njit(parallel=True)
def NLM (imagen, img_padding, dim_parche, h_cuadrado):
    '''
    Filtro Non-Local Means.
    
    Parámetros:
        imagen : matriz de píxeles normalizada con ruido. 
        img_padding : matriz de píxeles con padding.
        dim_parche : dimensión del parche que se necesita a la hora de filtrar. Para esta práctica 3. 
        h_cuadrado : parámetro de similitud asociado al grado de filtrado que se desea aplicar. 
        
    Devuelve:
        img_nlm : matriz de píxeles de la imagen filtrada. 
    
    Notas:
        Se filtra la imagen mediante un promedio ponderado de los píxeles de la imagen con ruido en
        función de la similitud de cada uno de los píxeles con lo demás.
    '''
    
    img_nlm = np.ones(imagen.shape) #Matriz de 1. 

    for i in range(1,img_padding.shape[0]-1): #Itera la matriz de la imagen con padding - Filas.
        for j in range(1,img_padding.shape[1]-1): #Itera la matriz de la imagen con padding - Columnas.
            parche_fijo = img_padding[i-1:i+dim_parche-1, j-1:j+dim_parche-1] #Parche de 3x3 (en nuestro caso).
            
            dist_array = np.ones(imagen.shape) #Matriz de distancias.
        
            for a in range(1,img_padding.shape[0]-1): #Itera la matriz de la imagen - Filas
                for b in range(1,img_padding.shape[1]-1): #Itera la matriz de la imagen - Columnas
                
                    parche_movil = img_padding[a-1:a+dim_parche-1, b-1:b+dim_parche-1] #Parche de 3x3 (en nuestro caso) 
                    resta_parches = (parche_fijo - parche_movil)**2 #Matriz 3x3 (en nuestro caso) 
                    dist_euclidea = np.sqrt(np.sum(resta_parches)) #Valor de la distancia de cada parche fijo al móvil. 
                    dist_array[a-1][b-1] = dist_euclidea #Por el padding. 
            
            dist_array[i-1][j-1] = 1 #Distancia alta para que salga un peso bajo cuando estamos en la posición del parche fijo.
            w = np.exp(-(dist_array)/(h_cuadrado)) 
            z_coef = np.sum(w) #Valor para cada parche fijo. 
        
            peso = (1/z_coef)*w #Matriz de pesos. 
        
            pixel_filtrado = np.sum(peso * imagen) #Se multiplica el pixel por cada ponderación y se suma.
            img_nlm[i-1][j-1] = pixel_filtrado #Por el padding. 
    
    return img_nlm


#NON LOCAL MEANS - CPP
@njit(parallel=True)
def NLM_CPP (imagen, img_padding, dim_parche, h_cuadrado, D0, alpha):
    '''
    Filtro Non-Local Means con modificación CPP.
    
    Parámetros:
        imagen : matriz de píxeles normalizada con ruido. 
        img_padding : matriz de píxeles con padding.
        dim_parche : dimensión del parche que se necesita a la hora de filtrar. Para esta práctica 3. 
        h_cuadrado : parámetro de similitud asociado al grado de filtrado que se desea aplicar. 
        Do : Parámetro de filtrado. 
        alpha : Parámetro de filtrado. 
        
    Devuelve:
        img_nlm_cpp : matriz de píxeles de la imagen filtrada NLM con la modificación CPP. 
    
    Notas:
        Se filtra la imagen mediante un promedio ponderado de los píxeles de la imagen con ruido en
        función de la similitud de cada uno de los píxeles con lo demás. Como se devuelve una imagen muy
        suavizada, con la modificación CPP se ponderarán los pesos originales del NLM para que dependan 
        también de la similitud entre píxeles centrales. 
    '''

    img_nlm_cpp = np.zeros(imagen.shape) #Matriz de 1. 

    for i in range(1,img_padding.shape[0]-1): #Itera la matriz de la imagen con padding - Filas.
        for j in range(1,img_padding.shape[1]-1): #Itera la matriz de la imagen con padding - Columnas.
            parche_fijo = img_padding[i-1:i+dim_parche-1, j-1:j+dim_parche-1] #Parche de 3x3 fijo.
            pixel_fijo = img_padding[i-1][j-1] #Píxel central fijo. 
            
            dist_array = np.zeros(imagen.shape) #Matriz de distancias para cada fijo. 
            resta_pixeles_centrados = np.zeros(imagen.shape)#Matriz de 0. 
            
            for a in range(1,img_padding.shape[0]-1): #Itera la matriz de la imagen con padding - Filas.
                for b in range(1,img_padding.shape[1]-1): #Itera la matriz de la imagen con padding  - Columnas.
                    parche_movil = img_padding[a-1:a+dim_parche-1, b-1:b+dim_parche-1] #Parche de 3x3 móvil. 
                    pixel_movil = img_padding[a-1][b-1] #Píxel central móvil.
                    
                    resta_pixeles_centrados [a-1][b-1] = np.abs(pixel_fijo - pixel_movil)
                    resta_parches = (parche_fijo - parche_movil)**2 #Matriz 3x3.
                    
                    dist_euclidea = np.sqrt(np.sum(resta_parches)) #Valor para cada parche móvil. 
                    dist_array[a-1][b-1] = dist_euclidea #Matriz de 64 valores para cada parche moóvil. 
            
            dist_array[i-1][j-1] = 1 #Distancia alta para que salga un peso bajo cuando estamos en la posición del parche fijo.
           
            n = 1/(1+(resta_pixeles_centrados/D0)**(2*alpha)) 
            w = np.exp(-(dist_array)/(h_cuadrado))
            z_coef = np.sum(w) #Valor para cada parche fijo.   
        
            peso = (1/z_coef)*w
            peso_cpp = peso*n #Nuevo peso para cada píxel.
            peso_cpp_norm = peso_cpp/np.sum(peso_cpp)#Normalización de los pesos. 
            
            img_nlm_cpp[i-1][j-1] = np.sum(peso_cpp_norm*imagen)#Se multiplica el pixel por cada ponderación y se suma.
    
    return img_nlm_cpp


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