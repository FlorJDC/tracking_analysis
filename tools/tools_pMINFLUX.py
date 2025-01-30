#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:53:34 2020

@author: Luciano Masullo, Lucía López and Lars Richter
I added a couple of functions. I noticed that `get_PSF` needs to be improved.  
TODO: fix this function to ensure it works properly and that its coupling with `analise_EBP`, which analyzes the pattern during alignment.
Florencia D. Choque 
"""
import matplotlib.pyplot as plt
from PIL import Image

from scipy import optimize as opt
import numpy as np
import glob
import os
import shutil
from skimage import io
from scipy import ndimage as ndi
import re

π = np.pi

#     define polynomial function to fit PSFs
def poly_func(grid, x0, y0, c00, c01, c02, c03, c04, c10, c11, c12, c13, c14, 
              c20, c21, c22, c23, c24, c30, c31, c32, c33, c34,
              c40, c41, c42, c43, c44):
    
    """    
    Polynomial function to fit PSFs.
    Uses built-in function polyval2d.
    
    Inputs
    ----------
    grid : x,y array
    cij : coefficients of the polynomial function
    Returns
    -------
    q : polynomial evaluated in grid.
    
    """

    x, y = grid
    c = np.array([[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14], 
                  [c20, c21, c22, c23, c24], [c30, c31, c32, c33, c34],
                  [c40, c41, c42, c43, c44]])
    q = np.polynomial.polynomial.polyval2d((x - x0), (y - y0), c)

    return q.ravel()


def spaceToIndex(space, size_nm, px_nm):

    # size and px have to be in nm
    index = np.zeros(2)
    index[0] = (size_nm/2 - space[1])/px_nm
    index[1] = (space[0]+ size_nm/2)/px_nm 

    return np.array(index, dtype=np.int)

def indexToSpace(index, size_nm, px_nm):

    space = np.zeros(2)
    space[0] = index[1]*px_nm - size_nm/2 # -size_nm/2 desplaza el origen de las coordenadas al centro de la imagen (en lugar de estar en la esquina superior izquierda).
    space[1] = size_nm/2 - index[0]*px_nm #Desplaza al centro de la imagen pero quisiera que pueda pasar no solo al centro de la imagen sino al centro de la dona cero
    #size_nm / 2 - # invierte el eje vertical, porque los índices de la matriz comienzan en la parte superior izquierda (donde las filas aumentan hacia abajo), mientras que en un gráfico cartesiano el eje y aumenta hacia arriba.
    return np.array(space)


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def getUniqueName(name):
    
    n = 1
    while os.path.exists(name + '.txt'):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1

    return name

    
def open_psf(filename, folder, subfolder):
    
    """   
    Open exp PSF images and drift data, fit exp data with poly_func
    Input
    ----------
    filename
    folder   
    subfolder 
    
    Returns
    -------
    
    PSF : (K, sizeg, sizeg) array, function from fit evaluated over grid
    x0, y0 : arrays, coordinates of EBP centers
    index : array, coordinates of EBP centers in indexes
    aopt : fit parameters, coefficients of polynomial function
       
    """
    
    # change dir to dir where PSF and drift data are located
    rootdir = r"C:\Users\Lucia\Documents\NanoFísica\MINFLUX\Mejor data TDIs"
    folder =  str(folder)
    subfolder = str(subfolder)
    newpath = os.path.join(rootdir, folder, subfolder)   
    os.chdir(newpath)
    
    # open any file with metadata from PSF images 
    fname = glob.glob('filename*')[0]   
    f = open(fname, "r")
    lines=f.readlines()
    # exp pixel size extracted from metadata
    pxexp = float(lines[8][22])
    print(pxexp)

    # open txt file with xy drift data
    c = str(filename) 
    cfile = str(c) + '_xydata.txt'
    coord = np.loadtxt(cfile, unpack=True)

    
    # open tiff stack with exp PSF images
    psffile = str(filename) + '.tiff'
    im = io.imread(psffile)
    imarray = np.array(im)
    psfexp = imarray.astype(float)
    
    # total number of frames
    frames = np.min(psfexp.shape)
    factor = 5.0
    # number of px in frame
    npx = np.size(psfexp, axis = 1)
    # final size of fitted PSF arrays (1 nm pixels)             
    sizepsf = int(factor*pxexp*npx)
        
    # number of frames per PSF (asumes same number of frames per PSF)
    fxpsf = frames//4

    # initial frame of each PSF
    fi = fxpsf*np.arange(5)  

    
    # interpolation to have 1 nm px and realignment with drift data
    psf = np.zeros((frames, sizepsf, sizepsf))        
    for i in np.arange(frames):
        psfz = ndi.interpolation.zoom(psfexp[i,:,:], factor*pxexp)    
        deltax = coord[1, i] - coord[1, 0]
        deltay = coord[2, i] - coord[2, 0]
        psf[i, :, :] = ndi.interpolation.shift(psfz, [deltax, deltay])

    # sum all interpolated and re-centered images for each PSF
    psfT = np.zeros((frames//fxpsf, sizepsf, sizepsf))
    for i in np.arange(4):
        psfT[i, :, :] = np.sum(psf[fi[i]:fi[i+1], :, :], axis = 0)
        
    # crop borders to avoid artifacts 
    w, h = psfT.shape[1:3]
    border = (w//5, h//5, w//5, h//5) # left, up, right, bottom
    psfTc = psfT[:, border[1]:h-border[1], border[0]:w-border[0]]
    psfTc = psfT[:, border[1]:h-border[1], border[0]:w-border[0]]
          
    # spatial grid
    sizeg = np.size(psfTc, axis = 1)
    sizexy = sizeg/factor
    pxg = 1/factor  # 1 nm px size for the function grid
    
    x = np.arange(0, sizexy, pxg)
    y = sizexy - np.arange(0, sizexy, pxg)
    x, y = np.meshgrid(x, y)
    
    # fit PSFs  with poly_func and find central coordinates (x0, y0)
    PSF = np.zeros((4, sizeg, sizeg))
    x0 = np.zeros(4)
    y0 = np.zeros(4)
    index = np.zeros((4,2))
    aopt = np.zeros((4,27))
#    x0fit = np.zeros(4)
#    y0fit = np.zeros(4)
        
    for i in np.arange(4):
        # initial values for fit parameters x0,y0 and c00
        ind1 = np.unravel_index(np.argmin(psfTc[i, :, :], 
            axis=None), psfTc[i, :, :].shape)
        x0i = x[ind1]
        y0i = y[ind1]
        c00i = np.min(psfTc[i, :, :])        
        p0 = [x0i, y0i, c00i, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        aopt[i,:], cov = opt.curve_fit(poly_func, (x,y), psfTc[i, :, :].ravel(), p0 = p0)    
        q = poly_func((x,y), *aopt[i,:])   
        PSF[i, :, :] = np.reshape(q, (sizeg, sizeg))
        # find min value for each fitted function (EBP centers)
        ind = np.unravel_index(np.argmin(PSF[i, :, :], 
            axis=None), PSF[i, :, :].shape)
        
        x0[i] = x[ind]
        y0[i] = y[ind]        
        index[i,:] = ind
  
    
    return PSF, x0, y0, index, aopt

def open_tcspc(supfolder, folder, subfolder, name):
    
    """   
    Open exp TCSPC data
    Input
    ----------
    
    supfolder
    folder   
    subfolder 
    name
    
    Returns
    -------
    
    absTime : array, absTime tags of collected photons
    relTime : array, relTime tags of collected photon
    τ : times of EBP pulses
    
    """
    
    # change dir to dir where TCSPC data is located
    rootdir = r"C:\Users\Lucia\Documents\NanoFísica\MINFLUX"
    supfolder = str(supfolder)
    folder =  str(folder)
    subfolder = str(subfolder)
    
    
    newpath = os.path.join(rootdir, supfolder, folder, subfolder)   
    os.chdir(newpath)
    
    # open txt file with TCSPC data
    tcspcfile = str(name) + '.txt'
    coord = np.loadtxt(tcspcfile, unpack=True)
    absTime = coord[1, :]
    relTime = coord[0, :]
    
    globRes = 1e-3 # absTime resolution 
    timeRes = 1 # relTime resolution (it's already in ns)
    
    absTime = absTime * globRes
    relTime = relTime * timeRes
    
    # find EBP pulses times
    [y, bin_edges] = np.histogram(relTime, bins=100)
    x = np.ediff1d(bin_edges) + bin_edges[:-1]
    
    T = len(y)//4*np.arange(5)
    
    ind = np.zeros(4)
    
    for i in np.arange(4):        
        ind[i] = np.argmax(y[T[i]:T[i+1]]) + T[i]
            
    
    ind = ind.astype(int)
    τ = x[ind]
    
    return absTime, relTime, τ
    


        
def n_minflux(τ, relTime, a, b):
    
    """
    Photon collection in a MINFLUX experiment
    (n0, n1, n2, n3)
    
    Inputs
    ----------
    τ : array, times of EBP pulses (1, K)
    relTime : photon arrival times relative to sync (N)
    a : init of temporal window (in ns)
    b : the temporal window lenght (in ns)
    
    a,b can be adapted for different lifetimes
    Returns
    -------
    n : (1, K) array acquired photon collection.
    
    """
    K = 4
    # total number of detected photons
    N = np.shape(relTime)[0]

    # number of photons in each exposition
    n = np.zeros(K)    
    for i in np.arange(K):
        ti = τ[i] + a
        tf = τ[i] + a + b
        r = relTime[(relTime>ti) & (relTime<tf)]
        n[i] = np.size(r)
        
    return n 


def pos_minflux(n, PSF, SBR,step_nm):
    
    """    
    MINFLUX position estimator (using MLE)
    
    Inputs
    ----------
    n : acquired photon collection (K)
    PSF : array with EBP (K x size x size)
    SBR : estimated (exp) Signal to Bkgd Ratio
    Returns
    -------
    indrec : position estimator in index coordinates (MLE)
    pos_estimator : position estimator (MLE)
    Ltot : Likelihood function
    
    Parameters 
    ----------
    step_nm : grid step in nm
        
    """

    # step_nm = 0.2
       
    # number of beams in EBP
    K = np.shape(PSF)[0]
    # FOV size
    size = np.shape(PSF)[1] 
    
    normPSF = np.sum(PSF, axis = 0)
    
    # probabilitiy vector 
    p = np.zeros((K, size, size))

    for i in np.arange(K):        
        p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * (1/K)

    # likelihood function
    L = np.zeros((K,size, size))
    for i in np.arange(K):
        L[i, :, :] = n[i] * np.log(p[i, : , :])
        
    Ltot = np.sum(L, axis = 0)

    # maximum likelihood estimator for the position    
    indrec = np.unravel_index(np.argmax(Ltot, axis=None), Ltot.shape)
    pos_estimator = indexToSpace(indrec, size, step_nm)
    
    return indrec, pos_estimator, Ltot


def likelihood(K, PSF, n, λb, pos_nm, step_nm, size_nm):
    
    """
    Computes the full likelihood for a given MINFLUX experiment 
    
    Input
    ----------
    K : int, number of excitation beams
    PSF : (K, size, size) array, experimental or simulated PSF
    n :  (1, K) array , photon collection 
    λb : float, bkgd level
    pos _nm : (K, 2) array, centers of the EBP positions
    step_nm : step of the grid in nm
    size_nm : size of the grid in nm
    Returns
    -------
    Like : (size, size) array, Likelihood function in each position
    
    """
    
    # size of the (x,y) grid
    size = int(size_nm/step_nm)
    
    # different arrays
    mle = np.zeros((size, size, K))
    p_array = np.zeros((size, size, K))
    λ_array = np.zeros((size, size, K))
    λb_array = np.ones((size, size)) * λb

    
    # λs in each (x,y)
    for i in np.arange(K):
        λ_array[:, :, i] = PSF[i, :, :]        
    
    norm_array = (K*λb + np.sum(λ_array, axis=2))
        
    # probabilities in each (x,y)
    for i in np.arange(K):
        p_array[:, :, i] = (λ_array[:, :, i] + λb_array)/norm_array
    
    # Likelihood
    for i in np.arange(K):
        mle[:, :, i] = n[i] * np.log(p_array[:, :, i])
        
    Like = np.sum(mle, axis = 2)
        
    return Like


def get_PSF(folder_path, K):
    """
    Recibe la dirección donde se encuentran las imágenes de las donas, las fitea, muestra la comparación.
    Grafica la posición de los mínimos encontrados de tres formas diferentes. (Decidí usar el min numérico para representar el centro del fit)
    Permite guardar las imágenes fiteadas en la carpeta 'fit' creada en el directorio donde están las imagenes de las PSFs.
    Este script obtiene las coordenadas de estos mínimos y grafica las posiciones de los mínimos.
    Además grafica las coord. X e Y por separado. FC.
    """
    file_paths = glob.glob(os.path.join(folder_path, '*.tif*')) # Cualquier extensión con sufijo tif es válida
    fit_directory = os.path.join(folder_path, 'fit')
    os.makedirs(fit_directory, exist_ok=True)
    
    psf_matrices = []
    pixel_sizes = []
    PSFs_min_num = []
    PSFs_min = []
    PSFs_min_x = []
    PSFs_min_y = []
    
    if len(file_paths) == 1: #Si hay un archivo TIFF, se asume que es un stack, y se extraen todas las img
        file_path = file_paths[0]
        psf_matrices = open_stack(file_path)
        pixel_size = search_px_size(file_path)
        pixel_sizes = [pixel_size]*len(psf_matrices)
    else: #Si hay varios archivos TIFF, cada uno se trata como una imagen individual
        for file_path in file_paths:
            imagen = open_stack(file_path)
            psf_matrices.extend(imagen) 
            pixel_size = search_px_size(file_path)
            pixel_sizes.append(pixel_size)
    j=0

    npx = np.size(psf_matrices[j], axis=1) # Numero de px de la imagen cruda, generalmente trabajo con 80px

    px_exp = pixel_size*1000 #tamaño del px en nm
    size_nm = npx*px_exp

    # interpolation
    #psfz = ndi.interpolation.zoom(averaged_psf_matrices[j], npx)
    
    # spatial grid
    x_like = np.arange(0, npx, 1/(pixel_size*1000)) #Si son 80 px de 6.25nm. Va de 0 a 80 en pasos de 1/6.25nm = 0.16 (nm-1)
    y_like = np.arange(0, npx, 1/(pixel_size*1000))
    likelihood_pix = np.size(x_like)
    x_like, y_like = np.meshgrid(x_like, y_like)

    
    x = np.arange(0, npx, 1)
    #print("[PSF_donut] x: ", x)
    y = np.arange(0, npx, 1)
    #print("[PSF_donut] y: ", y)
    x, y = np.meshgrid(x, y)
    #print("[PSF_donut] x after meshgrid: ", x)
    #print("[PSF_donut] y after meshgrid: ", y)

    # fit PSFs  with poly_func and find central coordinates (x0, y0)
    PSF = np.zeros((K, likelihood_pix, likelihood_pix))
    #PSF = np.zeros((K, npx, npx))
    x0 = np.zeros(K)
    y0 = np.zeros(K)
    # x0_min = np.zeros(K)
    # y0_min = np.zeros(K)
    #index = np.zeros((K,2))
    aopt = np.zeros((K,27))
    

    
    for j in np.arange(K): # 4 veces en p-minflux
        # Fila y columna del valor mínimo en la región
        index_min = np.unravel_index(np.argmin(psf_matrices[j], axis=None), psf_matrices[j].shape)
        x0i = x[index_min]
        y0i = y[index_min]
        print("y0i: ", y0i)
        
        # Valor mínimo de la img cruda
        c00i = np.min(psf_matrices[j])
        # Coordenadas del valor mínimo de la img cruda
        min_coords = np.where(psf_matrices[j] == c00i) #min coords:  <class 'tuple'> 
        #print(f"min coords dona {j}: {min_coords}")
        # Las coordenadas devueltas por np.where son tuplas de arrays (filas, columnas) 
        min_coords = list(zip(min_coords[0], min_coords[1]))
        PSFs_min.append(min_coords)
        
        p0 = [x0i, y0i, c00i, 0 ,1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        aopt[j,:], cov = opt.curve_fit(poly_func, (x,y), psf_matrices[j].ravel(), p0)
        q = poly_func((x_like,y_like), *aopt[j,:])
        
        PSF[j, :, :] = np.reshape(q, (likelihood_pix, likelihood_pix)) #PSF es la dona fiteada
        ## Dona fiteada
        minimo_fit = np.min(PSF[j]) #Minimo de la dona fiteada
        # Coordenadas del valor mínimo
        min_coords_fit = np.where(PSF[j] == minimo_fit) #min_coords_fit:  <class 'numpy.ndarray'>
        # Las coordenadas devueltas por np.where que asigno a la variable min_coords_fit son tuplas de arrays (filas, columnas)
        min_coords_fit = list(zip(min_coords_fit[0], min_coords_fit[1])) #type(min_coords_fit) = <class 'list'>
        PSFs_min_num.append(min_coords_fit)

        x0[j] = aopt[j,0] 
        y0[j] = aopt[j,1]
        x0[j] = np.round(x0[j]*pixel_size*1000,1) #type(x0): <class 'numpy.ndarray'>
        y0[j] = np.round(y0[j]*pixel_size*1000,1)
        PSFs_min_x.append(x0[j])
        PSFs_min_y.append(y0[j])
              
       
    #Transformo a array para trabajar más cómodamente
    PSFs_min_array = np.array(PSFs_min_num) 
    # print("np.shape(PSFs_min_array):", np.shape(PSFs_min_array))
    #Este orden es correcto, x es el 1, y es el 0
    x_donut = np.reshape(PSFs_min_array[np.arange(0,K,1),:,1],(K,)) # Son los x de la dona
    y_donut = np.reshape(PSFs_min_array[np.arange(0,K,1),:,0],(K,)) # coord y de la dona

    return PSF , x0, y0, PSFs_min_array , x_donut, y_donut

def promediar_tiff(tiff_file, K):
    """
    Parameters
    ----------
    tiff_file : Recibe el archivo tal cual sale del setup
    K : type(K):int. En general es 4. Sólo cambia cuando quiero analizar las donas por separado.
    
    Returns
    -------
    new_folder : Donde se guarda la imagen promedio.
    """
    #Para guardar imagen
    parent_directory = os.path.dirname(tiff_file)
    base_name, ext = os.path.splitext(os.path.basename(tiff_file))
    
    new_folder = os.path.join(parent_directory, f'{base_name}_Resultados')
    os.makedirs(new_folder, exist_ok=True)
    
    txt_file = os.path.join(parent_directory, base_name + ".txt")
    new_tiff_file = os.path.join(new_folder, base_name + "_promedio" + ext)
    new_txt_file = os.path.join(new_folder, base_name + "_promedio.txt")
    
    with Image.open(tiff_file) as img:
        num_images = img.n_frames 
        img_evento = num_images//K
        promedios = []
        for evento in range(K):
            suma = None
            for i in range(img_evento):
                img.seek(evento * img_evento + i)
                img_array = np.array(img, dtype=np.float32) 
                if suma is None:
                    suma = img_array
                else:
                    suma += img_array
            promedio = suma/img_evento
            promedios.append(promedio)
    
    new_img_list = [Image.fromarray(promedio.astype(np.float32), mode='F') for promedio in promedios] #'F': Imágenes en escala de grises de 32 bits por píxel (punto flotante)
    new_img_list[0].save(new_tiff_file, save_all=True, append_images=new_img_list[1:])
    shutil.copy(txt_file, new_txt_file)
    return new_folder


def open_stack(file_path):
    '''
    Esta función maneja tanto un archivo TIFF con múltiples frames (stack) como imágenes individuales,
    devolviendo siempre una lista de matrices.

    Parameters
    ----------
    file_path : Dirección de archivo tiff o tif.

    Returns
    -------
    psf_matrices : Lista de numpy arrays

    '''
    psf_matrices = []
    with Image.open(file_path) as tiff_stack:
        num_frames = tiff_stack.n_frames  # Número de imágenes en el stack
        for i in range(num_frames):
            tiff_stack.seek(i) #Selecciono el frame i
            imagen = np.array(tiff_stack, dtype=np.float32)
            psf_matrices.append(imagen)
    return psf_matrices

def search_px_size(file_path):
    txt_file_path = os.path.splitext(file_path)[0] + '.txt'
    with open(txt_file_path, 'r') as txt_file:
        txt_content = txt_file.read()
        pixel_size_match = re.search(r'pixel size \(µm\) = ([\d.]+)', txt_content)
        if pixel_size_match:
            return float(pixel_size_match.group(1))
        else:
            return None  # Si no se encuentra el valor, devolver None



def crb_minflux(K, PSF, SBR, px_nm, size_nm, N, method='1'):
    
    """
    
    Cramer-Rao Bound for a given MINFLUX experiment 
    
    Input
    ----------
    K : int, number of excitation beams
    PSF : (K, size, size) array, experimental or simulated PSF 
    SBR : float, signal to background ratio
    px_nm : pixel of the grid in nm
    size_nm : size of the grid in nm
    N : total number of photons
    method: parameter for the chosen method
    
    There are three methods to calculate it. They should be equivalent but
    provide different outputs.
    
    Method 1: calculates the σ_CRB using the most analytical result 
    (S26, 10.1126/science.aak9913)
    
    Output 1: σ_CRB (size, size) array, mean of CRB eigenval in each position
    
    Method 2: calculates the Σ_CRB from the Fisher information matrix in 
    emitter position space (Fr), from there it calculates Σ_CRB and σ_CRB
    (S11-13, 10.1126/science.aak9913)
    
    Output 2: Fr, Σ_CRB, σ_CRB
    
    Method 3: calculates the Fisher information matrix in reduced probability
    space and calculates J jacobian transformation matrices. From there it
    calculates Fr, Σ_CRB, σ_CRB. Fp, Σ_CRB_p and σ_CRB_p are additional outputs
    (S8-10, 10.1126/science.aak9913)
    
    Output 3: Fr, Σ_CRB, σ_CRB, Fp, Σ_CRB_p, σ_CRB_p
    """
    
    # size of the σ_CRB matrix in px and dimension d=2
    size = int(size_nm/px_nm)
    d = 2
    
    # size of the (x,y) grid
    dx = px_nm
    dy = px_nm
    
    if method=='1':
                
        # define different arrays needed to compute CR
        
        p, λ, dpdx, dpdy, A, B, C, D = (np.zeros((K, size, size)) for i in range(8))
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))

        for i in range(K):
    
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
            
        # λb using the approximation in Balzarotti et al, (S29)
            
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(K*SBR)
        
        # probabilities in each (x,y)
        
        for i in np.arange(K):
            
            # probability arrays
    
            p[i, :, :] = (λ[i, :, :] + λb)/(K*λb + np.sum(λ, axis=0))
            
            # plot of p
            
            locx = (size/4) * np.sqrt(2)/2
            locy = (size/4) * np.sqrt(2)/2
            
            #plt.figure(str(i))
            #plt.plot(np.arange(-size/2, size/2), p[i, int(size/2 - locx), :], label='p x axis')
            #plt.plot(np.arange(-size/2, size/2), p[i, ::-1, int(size/2 - locy)], label='p y axis')
                                    
            # gradient of ps in each (x,y)
            
            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
           
            # terms needed to compute CR bound in aeach (x,y)
            
            A[i, :, :] = (1/p[i, :, :]) * dpdx[i, :, :]**2
            B[i, :, :] = (1/p[i, :, :]) * dpdy[i, :, :]**2
            C[i, :, :] = (1/p[i, :, :]) *(dpdx[i, :, :] * dpdy[i, :, :])
            D[i, :, :] = (1/p[i, :, :]) * (dpdx[i, :, :]**2 + dpdy[i, :, :]**2)
    
        # sigma Cramer-Rao numerator and denominator    
        E = np.sum(D, axis=0) 
        F = (np.sum(A, axis=0) * np.sum(B, axis=0)) - np.sum(C, axis=0)**2
        
        σ_CRB = np.sqrt(1/(d*N))*np.sqrt(E/F)
            
        return σ_CRB
    
    if method=='2':
    
        # initialize different arrays needed to compute σ_CRB, Σ_CRB and Fr
        
        σ_CRB = np.zeros((size, size))
        p, λ, dpdx, dpdy = (np.zeros((K, size, size)) for i in range(4))
        Fr, Σ_CRB = (np.zeros((d, d, size, size)) for i in range(2))
        
        Fr_aux = np.zeros((K, d, d, size, size))
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))

        for i in range(K):
            
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
            
        # λb using the approximation in Balzarotti et al, (S29)
          
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(K*SBR)
            
        for i in range(K):
            
            # probability arrays
        
            p[i, :, :] = (λ[i, :, :] + λb)/(K*λb + np.sum(λ, axis=0))

            # partial derivatives in x and y direction
    
            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
            
        # compute relevant information for every (i, j) position
        # TODO: vectorize this part of the code
            
        for i in range(size):
            for j in range(size):
                
                for k in range(K):
            
                    A = np.array([[dpdx[k, i, j]**2, 
                                   dpdx[k, i, j]*dpdy[k, i, j]],
                                  [dpdx[k, i, j]*dpdy[k, i, j], 
                                   dpdy[k, i, j]**2]])
        
                    Fr_aux[k, :, :, i, j] = (1/p[k, i, j]) * A
                    
                Fr[:, :, i, j] = N * np.sum(Fr_aux[:, :, :, i, j], axis=0)
                                    
                Σ_CRB[:, :, i, j] = np.linalg.inv(Fr[:, :, i, j])
                σ_CRB[i, j] = np.sqrt((1/d) * np.trace(Σ_CRB[:, :, i, j]))
                
        
        return σ_CRB, Σ_CRB, Fr
            
         
    if method=='3':
    
        # initalize σ_CRB and E(logL)
        
        I_f = np.zeros((size, size))
        σ_CRB = np.zeros((size, size))
        σ_CRB2 = np.zeros((size, size))
        σ_CRB_p = np.zeros((size, size))
        
        logL = np.zeros((size, size))
    
        # initialize different arrays needed to compute σ_CRB, Σ_CRB, Fr, etc
    
        p, λ, dpdx, dpdy, logL_aux = (np.zeros((K, size, size)) for i in range(5))
        Fr, Σ_CRB = (np.zeros((d, d, size, size)) for i in range(2))
        Fp, Σ_CRB_p = (np.zeros((K-1, K-1, size, size)) for i in range(2))
    
        J = np.zeros((K-1, d, size, size))
        
        diag_aux = np.zeros(K-1)
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))
                      
        for i in range(K):
            
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
                    
        # λb using the approximation in Balzarotti et al, (S29)
              
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(K*SBR)
            
        for i in range(K):
            
            # probability arrays
            
            p[i, :, :] = (λ[i, :, :] + λb)/(K*λb + np.sum(λ, axis=0))
            
            logL_aux[i, :, :] = N * p[i, :, :] * np.log(p[i, :, :])
            
            # partial derivatives in x and y direction

            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
            
#            locx = (size/4) * np.sqrt(2)/2
#            locy = (size/4) * np.sqrt(2)/2
            
            locx = (size/4)
            locy = (size/4)
#            
            plt.figure(str(i))
            plt.imshow(p[i, :, :])
            
            plt.figure()
            plt.plot(np.arange(-size/2, size/2), p[i, int(size/2), :], label='p x axis')
            plt.plot(np.arange(-size/2, size/2), p[i, ::-1, int(size/2 - locy)], label='p y axis')
            plt.legend()   
            
            
        for i in range(size):
            for j in range(size):
                    
                for k in range(K):
                    
                    if k < K-1:
                        
                        J[k, :, i, j] = np.array([dpdx[k, i, j], dpdy[k, i, j]])
                        
#                    if k < K-2:
                        
                        diag_aux[k] = 1/p[k, i, j]
                        
                    else:
                        
                        pass
                    
                logL[i, j] = np.sum(logL_aux[:, i, j], axis=0)
                        
                p_aux = np.diag(diag_aux)
                
                Fp[:, :, i, j] = N * (p_aux + np.ones((K-1, K-1))*(1/p[K-1, i, j]))
                Fr[:, :, i, j] = J[:, :, i, j].T.dot(Fp[:, :, i, j]).dot(J[:, :, i, j])
                                
                Σ_CRB[:, :, i, j] = np.linalg.inv(Fr[:, :, i, j])
                σ_CRB[i, j] = np.sqrt((1/d) * np.trace(Σ_CRB[:, :, i, j]))
                
                Σ_CRB_p[:, :, i, j] = np.linalg.inv(Fp[:, :, i, j])
                σ_CRB_p[i, j] = np.sqrt((1/(K-1)) * np.trace(Σ_CRB_p[:, :, i, j]))
                
                I_f[i, j] = np.sqrt((1/d) * np.trace(Fr[:, :, i, j]))
                σ_CRB2[i, j] = 1/I_f[i, j] 

                
        print(Fr[:, :, int(size/2), int(size/2 - locx)])
        print(Fr[:, :, int(size/2 + locy), int(size/2)])
        print(Fr[:, :, int(size/2), int(size/2 + locy)])
        print(Fr[:, :, int(size/2 - locy), int(size/2)])
        
                
        
#        print(p[:, int(size/2 - locy), int(size/2 - locx)])
#        print(p[:, int(size/2 + locy), int(size/2 - locx)])
#        print(p[:, int(size/2 - locy), int(size/2 + locx)])
#        print(p[:, int(size/2 + locy), int(size/2 + locx)])
        
        print(p[:, int(size/2), int(size/2)])
        
        print(p[:, int(size/2), int(size/2 - locx)])
        print(p[:, int(size/2 + locy), int(size/2)])
        print(p[:, int(size/2), int(size/2 + locy)])
        print(p[:, int(size/2 - locy), int(size/2)])
        
        print(σ_CRB[int(size/2), int(size/2 - locx)])
        print(σ_CRB[int(size/2 + locy), int(size/2)])
        print(σ_CRB[int(size/2), int(size/2 + locy)])
        print(σ_CRB[int(size/2 - locy), int(size/2)])


        return σ_CRB, Σ_CRB, Fr, σ_CRB_p, Σ_CRB_p, Fp, logL, I_f, σ_CRB2
    
    else:
        
        raise ValueError('Invalid method number, please choose 1, 2 or 3 \
                         according to the desired calculation')
     
