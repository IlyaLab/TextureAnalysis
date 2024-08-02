# -*- coding: utf-8 -*-
"""
Taken from the scikit-image library and modified.
Stores all of the GLCM Feature equations, both the ones already in skimage and the ones added based off of the Hao et al. and Trace papers.
"""

import numpy as np
import scipy as sp
from skimage._shared.utils import check_nD
import skimage.feature

def graycoprops(P, prop='contrast'):
    
    check_nD(P, 4, 'P')

    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1. / (1. + (I - J) ** 2)
    elif prop == 'autocorrelation': # ADDED
        weights = I * J
    elif prop == 'NID': # Normed Inverse Difference     # ADDED
        weights = 1. / (1. + (np.abs(I - J) / num_level ** 2) )
    elif prop == 'NIM': # Normed Inverse Difference Moment  # ADDED
        weights = 1. / (1. + ( ((I - J) ** 2) / num_level ** 2) )
    elif prop in ['ASM', 'energy', 'correlation', 'cluster prominence',
                  'cluster shade', 'entropy', 'max probability', 'sum of squares',
                  'sum average', 'sum variance', 'sum entropy', 'difference variance',
                  'difference entropy', 'trace', 'information measures 1', 'information measures 2']:
        pass
    else:
        raise ValueError(f'{prop} is an invalid property')

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.sum(P ** 2, axis=(0, 1))
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.sum(P ** 2, axis=(0, 1))
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.sum(I * P, axis=(0, 1))
        diff_j = J - np.sum(J * P, axis=(0, 1))

        std_i = np.sqrt(np.sum(P * (diff_i) ** 2, axis=(0, 1)))
        std_j = np.sqrt(np.sum(P * (diff_j) ** 2, axis=(0, 1)))
        cov = np.sum(P * (diff_i * diff_j), axis=(0, 1))

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
        
    elif prop == 'cluster prominence':  # ADDED
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        u_x = np.sum(I * P, axis=(0, 1))
        u_y = np.sum(J * P, axis=(0, 1))
        results = np.sum(((I + J - u_x - u_y) ** 4) * P, axis=(0, 1))
        
    elif prop == 'cluster shade':   # ADDED
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        u_x = np.sum(I * P, axis=(0, 1))
        u_y = np.sum(J * P, axis=(0, 1))
        results = np.sum(((I + J - u_x - u_y) ** 3) * P, axis=(0, 1))
        
    elif prop == 'entropy':     # ADDED
        results = sp.stats.entropy(P, base=10, axis=(0, 1))
            
    elif prop == 'max probability':     # ADDED
        # results = np.fmax(P)
        results = np.zeros([1,num_angle], dtype=object)
        for theta in range(num_angle):
            results[0,theta] = P[:,:,0,theta].max()
            
    elif prop == 'sum of squares':      # ADDED
        results = np.zeros([1,num_angle])
        u = np.mean(P)
        for theta in range(num_angle):
            results[0,theta] = np.sum( (I - u) ** 2 * P[:,:,0,theta] )
        
    elif prop == 'sum average':     # ADDED   
        ans = np.sum(P, axis=(0,1))
        results = np.sum(k * ans for k in range(num_level*2))

    elif prop == 'sum variance':        # ADDED        
        ans = np.sum(P, axis=(0,1))
        senth = - np.sum(ans * np.log10(ans) for k in range(num_level*2))
        results = np.sum((k - senth)**2 * ans for k in range(num_level*2))

    elif prop == 'sum entropy':     # ADDED
        ans = np.sum(P, axis=(0,1))
        results = - np.sum(ans * np.log10(ans) for k in range(num_level*2))

    elif prop == 'difference variance':     # ADDED
        ans = np.sum(P, axis=(0,1))
        results = np.sum(k**2 * ans for k in range(num_level-1))
        
    elif prop == 'difference entropy':      # ADDED
        ans = np.sum(P, axis=(0,1))
        results = - np.sum(ans * np.log10(ans) for k in range(num_level-1))
        
    elif prop == 'trace':   # ADDED
        results = np.trace(P)
               
    elif prop in ['contrast', 'dissimilarity', 'homogeneity', 'autocorrelation',
                  'NID', 'NIM']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum(P * weights, axis=(0, 1))

    return results




def co_occurrence_features(p):
    # Calculate GLCM features (skimage)
    contrast = skimage.feature.graycoprops(p,prop='contrast')
    correlation = skimage.feature.graycoprops(p,prop='correlation')
    dissimilarity = skimage.feature.graycoprops(p,prop='dissimilarity')
    energy = skimage.feature.graycoprops(p,prop='energy')
    homogeneity = skimage.feature.graycoprops(p,prop='homogeneity')
    ASM = skimage.feature.graycoprops(p,prop='ASM')
    
    # Calculate GLCM features (added from Hao et al.)
    autoc = graycoprops(p,prop='autocorrelation')
    cprom = graycoprops(p,prop='cluster prominence')
    cshade = graycoprops(p,prop='cluster shade')
    entropy = graycoprops(p,prop='entropy')
    maxprob = graycoprops(p,prop='max probability')
    sumsq = graycoprops(p,prop='sum of squares')
    sumavg = graycoprops(p,prop='sum average')  # not included
    sumvar = graycoprops(p,prop='sum variance') # not included
    sumentr = graycoprops(p,prop='sum entropy')
    diffvar = graycoprops(p,prop='difference variance')
    diffentr = graycoprops(p,prop='difference entropy')
    NID = graycoprops(p,prop='NID')
    NIM = graycoprops(p,prop='NIM')

    # added from Trace paper
    trace = graycoprops(p, prop='trace')

    feature_vector = np.concatenate((contrast,correlation,dissimilarity,energy,
                                      homogeneity,ASM,autoc,cprom,cshade,entropy,maxprob,
                                      sumsq,sumavg,sumvar,sumentr,diffvar,diffentr,
                                      NID,NIM, trace),axis=0)

    return feature_vector




