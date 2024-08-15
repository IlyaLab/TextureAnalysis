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

# this is defined here to be used in ML.py for feature_importance
feature_headers = []
for channel in channel_names:
    for angle in angle_names:
        for feature in feature_names:
            feature_headers.append((feature + '-' + channel + '-' + angle))


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
            
    # finds the number of features through running feature extraction on the last tile in order to find other dimension for feature_matrix storage
    RGBImg = np.asarray(Img)
    feature_vector_RGB = calc_features(RGBImg, distances=distances, angles=angles)
    feature_vector = feature_vector_RGB

    # addition of HSV texture features
    hsv_img = Img.convert('HSV')
    hsv_img = np.array(hsv_img)
    feature_vector_HSV = calc_features(hsv_img, distances=distances, angles=angles)
    feature_vector = np.concatenate((feature_vector, feature_vector_HSV), axis=0)

    # addition of LAB texture freatures 
    lab_img = Img.convert('LAB')
    lab_img = np.array(lab_img)
    feature_vector_LAB = calc_features(lab_img, distances=distances, angles=angles)
    feature_vector = np.concatenate((feature_vector, feature_vector_LAB), axis=0)

    # builds feature_matrix
    feature_matrix = np.zeros([len(feature_vector),ultima_counter], dtype=object)

    del feature_vector, feature_vector_RGB, feature_vector_HSV, feature_vector_LAB, Img, RGBImg, hsv_img
    
    ultima_counter = 0

    for folder in os.listdir(Folders_Path):
        tile_counter = 0
        Tiles_Path = os.path.join(Folders_Path, folder)

        for file in os.listdir(Tiles_Path):
            time1 = time.time()
            tile_counter+=1
            ultima_counter+=1

            image_path = os.path.join(Tiles_Path, file)
            Img = Image.open(image_path) 
            RGBImg = np.asarray(Img)
            tile_name = 'Tile' + str(tile_counter)
            RGBImg, BG_check = normalizeStaining(RGBImg[:,:,0:3], tile_name, saveFile=tile_name)
            if BG_check == True:
                tile_counter-=1
                ultima_counter-=1
                continue

            feature_vector_RGB = calc_features(RGBImg, distances=distances, angles=angles)
            # all color channels concatenated
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
            RGBImg, tile_name, BG_check, hsv_img, 
            feature_vector_RGB, feature_vector_HSV, feature_vector_LAB, lab_img, image_path
            try:
                file_name = re.sub('.png$', '', file)
            except:
                pass
            try:
                file_name = file.re.sub('.tiff$', '', file)
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
    
    # create the columns for the data frame made from the feature_matrix array
    columns = []
    for channel in channel_names:
        for angle in angle_names:
            for feature in feature_names:
                columns.append((feature + '-' + channel + '-' + angle))

    feature_matrix = pd.DataFrame(feature_matrix, columns=columns)
    
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