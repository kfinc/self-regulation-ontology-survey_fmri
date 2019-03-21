
# coding: utf-8

# In[15]:

import os
import glob as glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import nibabel as nib
from nilearn import image, plotting
from nilearn.plotting import plot_stat_map, plot_anat, plot_img
from nilearn.image import load_img, new_img_like
from nilearn.input_data import NiftiMasker
from nilearn import datasets, plotting, input_data, signal, image
from nilearn.input_data import NiftiLabelsMasker

from secondlevel_utils import *

#--- setup

zmaps_dir = '/oak/stanford/groups/russpold/data/uh2/BIDS_data/derivatives/surveyMedley/1stlevel_surveyMedley/'
#loading Schaefer atlas (400 regions)
n_rois = 400
atlas_filename = '/home/users/finc/Self_Regulation_Ontology_Survey_fMRI/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm_resampled.nii.gz'
parcel = nib.load(atlas_filename)

subs_list = np.sort(os.listdir(zmaps_dir))
subs_list = [i for i in subs_list if len(i) == 8]

parcellation = 'shaefer_400_7'
n_trials = 40


for sub in subs_list:
    print('=========================================')
    print(f'RDMs calculation {sub}')
    print('=========================================')

    zmaps_path = f'{zmaps_dir}{sub}/{sub}_surveyMedley_Q_all_zmaps.nii.gz'

    if not os.path.exists(zmaps_path):
        print(f'No zmaps for {sub}')
        continue

    zmaps = nib.load(zmaps_path)


    rdms = np.zeros((n_rois, 40, 40))
    corr = np.zeros((n_rois, 40, 40))


    for i in range(n_rois):
        mask_vector = mask_map_files(roi_i = i, map_files = zmaps, parcel = parcel, extraction_dir = zmaps_dir,
                       metadata=None, labels=None, rerun=True,
                       threshold=0, save=False)
        cor = np.corrcoef(mask_vector)
        rdms[i,:,:] = 1 - cor
        corr[i,:,:] = cor

    np.save(f'{zmaps_dir}{sub}/{sub}_{parcellation}_RDMs', rdms)
    np.save(f'{zmaps_dir}{sub}/{sub}_{parcellation}_corr', corr)
    print('Done')
    print(' ')
