# -*- coding: utf-8 -*-
"""
Runs the GLCM feature extraction on each tile image and saves the results to a feature vector bank. 
"""

import os
import numpy as np
import pandas as pd
import skimage
from PIL import Image
import pandas as pd
import time
import re
from multiprocessing import Process
import warnings

import skimage.color

import GLCMFeatures as glcm
from Color_normalization import normalizeStaining

# to modify saved feature names, color channels, angles in data frame, or distances, change these lists
feature_names = ['Contrast', 'Correlation', 'Dissimilarity', 'Energy', 'Homogeneity', 'ASM', 'Autocorrelation', 'Cluster Prominence', 'Cluster Shade', 'Entropy', 
                     'Max Probability', 'Sum of Squares', 'Sum Average', 'Sum Variance', 'Sum Entropy', 'Difference Variance', 'Difference Entropy','NID', 'NIM', 
                     'Trace',]
channel_names = ['r', 'g', 'b', 'h', 's', 'v', 'L', 'A', 'B']
angle_names = ['0', '45', '90', '135']
distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# global feature_headers
feature_headers = [(feature + '-' + channel + '-' + angle) 
                    for channel in channel_names 
                    for angle in angle_names 
                    for feature in feature_names]

# value set to remove the background tiles that were incorrectly made by the grid_tiler()
contrast_threshold = 30

# CPU multiprocessing
def feature_extraction(src, save_path):
    processes = []
    i = 1
    for subset in os.listdir(src):
        processes.append(Process(target=texture_features, args=(os.path.join(src, subset), os.path.join(save_path, (str(i) + '.csv')) ), ) )
        i+=1
    for process in processes:
        process.start()
    for process in processes:
        process.join()


# checks for background tiles that were incorretly made by the grid_tiler()
def low_contrast_check(RGBimg):
    Img_1 = RGBimg[:,:,0]
    Img_2 = RGBimg[:,:,1]
    Img_3 = RGBimg[:,:,2]

    # creates the GLCM from each color channel
    p_1 = skimage.feature.graycomatrix(np.array(Img_1), distances, angles, normed=True)
    p_2 = skimage.feature.graycomatrix(np.array(Img_2), distances, angles, normed=True)
    p_3 = skimage.feature.graycomatrix(np.array(Img_3), distances, angles, normed=True)

    contrast_1 = glcm.graycoprops(p_1, prop='contrast')
    contrast_2 = glcm.graycoprops(p_2, prop='contrast')
    contrast_3 = glcm.graycoprops(p_3, prop='contrast')

    if np.any(contrast_1 <= contrast_threshold) or np.any(contrast_2 <= contrast_threshold) or np.any(contrast_3 <= contrast_threshold):
        return True
    else:
        return False


# main()
def texture_features(Folders_Path, save_path):
    if save_path.endswith('.csv') == True:
        pass
    else:
        try:
            save_path = os.path.join(save_path, 'texture_results.csv')
        except:
            raise ValueError('save_path must either be a .csv or directory.')

    ultima_counter = 0 #keeps track of total number of tiles
    index = []

    # finds the total number of tiles in order to construct a feature_matrix for storage
    for folder in os.listdir(Folders_Path):
        Tiles_Path = os.path.join(Folders_Path, folder)
        for file in os.listdir(Tiles_Path):
            if ultima_counter == 0:
                Img = Image.open(os.path.join(Tiles_Path, file))
            ultima_counter+=1

    # builds feature_matrix
    feature_matrix = np.zeros([len(feature_headers),ultima_counter], dtype=object)
    
    ultima_counter = 0

    for folder in os.listdir(Folders_Path):
        tile_counter = 0 # counts number of tiles per sample
        Tiles_Path = os.path.join(Folders_Path, folder)

        for file in os.listdir(Tiles_Path):
            time1 = time.time()
            tile_counter+=1
            ultima_counter+=1

            image_path = os.path.join(Tiles_Path, file)
            Img = Image.open(image_path) 
            Img_arr = np.asarray(Img)
            tile_name = 'Tile' + str(tile_counter)

            # Normalizes tile to H&E standard and first place to catch background tiles that made it through
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    RGBImg = normalizeStaining(Img_arr[:,:,0:3], tile_name, saveFile=tile_name)
                except RuntimeWarning:
                    tile_counter-=1
                    ultima_counter-=1
                    Img.close()
                    # drop the last empty column of the feature matrix of zeros
                    feature_matrix = feature_matrix[:,:-1]
                    del Img, tile_name, Img_arr, image_path
                    continue
            
            # second place to catch background tiles that made it through
            low_contrast = low_contrast_check(Img_arr[:,:,0:3])
            if low_contrast == True:
                tile_counter-=1
                ultima_counter-=1
                Img.close()
                # drop the last empty column of the feature matrix of zeros
                feature_matrix = feature_matrix[:,:-1]
                del Img, RGBImg, tile_name, Img_arr, image_path
                continue
            else:
                RGBImg = Img_arr[:,:,0:3]
            
                # Texture features of RGB channels
                feature_vector_RGB = calc_features(RGBImg, distances=distances, angles=angles)
                feature_vector = feature_vector_RGB

                # Texture features of HSV channels
                hsv_img = Img.convert('HSV')
                hsv_img = np.array(hsv_img)
                feature_vector_HSV = calc_features(hsv_img, distances=distances, angles=angles)
                feature_vector = np.concatenate((feature_vector, feature_vector_HSV), axis=0)

                # Texture features of LAB channels
                lab_img = Img.convert('LAB')
                lab_img = np.array(lab_img)
                feature_vector_LAB = calc_features(lab_img, distances=distances, angles=angles)
                feature_vector = np.concatenate((feature_vector, feature_vector_LAB), axis=0)

                # storing features within the matrix
                feature_matrix[:, ultima_counter-1] = feature_vector

                # frees up space by deleting variables
                Img.close()
                del feature_vector, Img, 
                RGBImg, tile_name, hsv_img, 
                feature_vector_RGB, feature_vector_HSV, feature_vector_LAB, lab_img, image_path
                
                # removes ext from file name before including it in dataframe
                try:
                    file_name = re.sub('.png$', '', file)
                except:
                    pass
                try:
                   file_name = re.sub('.tiff$', '', file)
                except:
                    pass
                
                try:
                    index.append(file_name)
                except:
                    index.append(file)

                elapsed_time = time.time() - time1
                print('\nTile ' + str(ultima_counter) + '\n' + str(elapsed_time))
        
    # convert to DataFrame with texture features
    feature_matrix = feature_matrix.T

    feature_matrix = pd.DataFrame(feature_matrix, columns=feature_headers)
    
    feature_matrix['Tile'] = index
    feature_matrix.set_index('Tile', inplace=True)
    feature_matrix.to_csv(save_path)


def calc_features(Img_arr, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    Img_1 = Img_arr[:,:,0]
    Img_2 = Img_arr[:,:,1]
    Img_3 = Img_arr[:,:,2]

    # creates the GLCM from each color channel
    p_1 = skimage.feature.graycomatrix(np.array(Img_1), distances, angles, normed=True)
    p_2 = skimage.feature.graycomatrix(np.array(Img_2), distances, angles, normed=True)
    p_3 = skimage.feature.graycomatrix(np.array(Img_3), distances, angles, normed=True)

    # calculates the co-occurrence features
    feature_vector_1 = glcm.co_occurrence_features(p_1)
    feature_vector_2 = glcm.co_occurrence_features(p_2)
    feature_vector_3 = glcm.co_occurrence_features(p_3)
                
    # all angles concatenated into one feature vector per color channel
    for col in range(len(feature_vector_1.T)):
        try:
            v1 = np.concatenate((v1, feature_vector_1[:,col]), axis=0)
            v2 = np.concatenate((v2, feature_vector_2[:,col]), axis=0)
            v3 = np.concatenate((v3, feature_vector_3[:,col]), axis=0)
        except:
            v1 = feature_vector_1[:,col]
            v2 = feature_vector_2[:,col]
            v3 = feature_vector_3[:,col]

    feature_vector = np.concatenate((v1, v2, v3), axis=0)

    return feature_vector
