# coding: utf-8

# In[8]:


import os
import glob as glob
import matplotlib.pyplot as plt
#$%matplotlib inline

import pandas as pd
import numpy as np

import nibabel as nib
from nilearn import image, plotting
from nistats.design_matrix import make_first_level_design_matrix
from nistats.reporting import plot_design_matrix,  plot_contrast_matrix
from nistats.first_level_model import FirstLevelModel
from nilearn.plotting import plot_stat_map, plot_anat, plot_img

from confounds_prep import *

#--- setup
beh_dir = '/home/users/finc/prep_order/'
out_dir = '/oak/stanford/groups/russpold/data/uh2/BIDS_data/derivatives/surveyMedley/1stlevel_surveyMedley/'
fmri_dir = '/oak/stanford/groups/russpold/data/uh2/BIDS_data/derivatives/fmriprep/fmriprep/'

task = 'surveyMedley'
subs_list = np.sort(os.listdir(fmri_dir))
subs_list = [i for i in subs_list if len(i) == 8]

tr = 0.68
n_trials = 40


for sub in subs_list:
    print('=========================================')
    print(f'Processing {sub}')
    print('=========================================')
    
    #--- loading fMRI data
    try :
        fmri = glob.glob(f'{fmri_dir}{sub}/*/func/*{task}*bold_space-MNI152NLin2009cAsym_preproc.nii.gz')[0]
        fmri_img = nib.load(fmri)
        n_scans = fmri_img.shape[3]

        new_dir = f'{out_dir}{sub}/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        os.makedirs(f'{new_dir}figures', exist_ok=True)

    except IndexError:
        with open(f'{out_dir}missing_fmri.txt', 'a') as missing_fmri:
            missing_fmri.write(f'{sub}\n')
            
        print(f'No fMRI data for {sub}')
        print('')
        continue
        
    #--- loading events data
    try :
        events_prep = pd.read_csv(f'{beh_dir}/{sub}_{task}_events.csv')
        frame_times = np.arange(n_scans) * tr
        events = pd.DataFrame(events_prep, columns=['onset', 'trial_type', 'duration'])
        events_path = f'{out_dir}{sub}/{sub}_{task}_events.csv'

        if not os.path.exists(events_path):
            events.to_csv(events_path, index = False)

        n = len(np.unique(events['trial_type']))
        assert n == n_trials,  f'Wrong number of qestions for {sub}. Should be {n_trials}, but is {n}'
        assert events.isnull().sum().values.sum() == 0,  f'Not a number values in events for {sub}.'

    except (IndexError, AssertionError) as e:
        with open(f'{out_dir}missing_events.txt', 'a') as missing_events:
            missing_events.write(f'{sub}\n')
        print(f'No events data for {sub}')
        print('')
        continue

    #--- loading confounds
    try :
        confounds = pd.read_csv(glob.glob(f'{fmri_dir}{sub}/*/func/*surveyMedley*.tsv')[0],  sep = '\t')
        confounds_clean = confounds_prep(confounds)                             

        #n_outliers = sum(confounds_clean['scrubbing'])
        #perc_ouliers = (n_outliers/n_scans)*100
        confounds_path = f'{out_dir}{sub}/{sub}_{task}_confounds_clean.csv'
        if not os.path.exists(confounds_path):
            confounds_clean.to_csv(confounds_path, index = False)

    except IndexError:
        with open(f'{out_dir}missing_confounds.txt', 'a') as missing_confounds:
            missing_confounds.write(f'{sub}\n')
            
        print(f'No confounds data for {sub}')
        print('')
        continue


    try :    
        fd = confounds['FramewiseDisplacement'].copy()
        mean_fd = fd.values[1:].mean(axis=0)

        #assert perc_ouliers < 15,  f'More than 15% outliers for {sub}'
        assert  mean_fd < 0.5,  f'Mean FD higher than 0.5 mm for {sub}'

    except  AssertionError:
        with open(f'{out_dir}high_motion.txt', 'a') as high_motion:
            high_motion.write(f'{sub}\n')
        
        
    #--- definng the model
    fmri_glm = FirstLevelModel(tr, 
                               noise_model='ar1',
                               standardize=False, 
                               hrf_model='spm',
                               drift_model='cosine',
                               period_cut=128,
                              )

    fmri_glm_non_smoothed = fmri_glm.fit(fmri, events, confounds=confounds_clean)
    design_matrix = fmri_glm_non_smoothed.design_matrices_[0]
    fig, ax = plt.subplots(figsize=(15,8))
    plot_design_matrix(design_matrix, ax=ax)
    fig.savefig(f'{out_dir}{sub}/figures/{sub}_design_matrix.png')
    
    #--- defining contrasts
    trial_dummies = pd.get_dummies(design_matrix.columns)
    trial_contrasts = pd.DataFrame.to_dict(trial_dummies, orient = 'list')
    trial_contrasts['overall'] = [1 if i[0] == 'Q' else 0 for i in design_matrix.columns]

    fig, ax = plt.subplots(n_trials, 1)
    fig.set_size_inches(22, 40)

    for i in range(n_trials ):
        plt.figure()
        _ = plot_contrast_matrix(trial_contrasts[f'Q{i+1:02}'] , design_matrix=design_matrix.iloc[:,0:40], ax=ax[i])

    fig.savefig(f'{out_dir}{sub}/figures/{sub}_contrasts_trials.png')
    
    #--- trial effects estimation
    zmaps = np.zeros((89, 105, 89, n_trials))

    for i in range(n_trials):
        zmap = fmri_glm_non_smoothed.compute_contrast(np.asarray(trial_contrasts[f'Q{i+1:02}']), output_type='z_score')
        zmaps[:,:,:,i] = zmap.get_fdata()
        p = plot_stat_map(zmap, threshold=3,
                  display_mode='z', cut_coords=8, black_bg=False,
                  title='Q{:0>2d}'.format(i+1))
        p.savefig(f'{out_dir}{sub}/figures/{sub}_Q{i+1:02}_zmap.png')

    zmap_img = nib.Nifti1Image(zmaps, zmap.affine, zmap.header)
    nib.save(zmap_img, f'{out_dir}{sub}/{sub}_{task}_Q_all_zmaps.nii.gz')
    
    #--- overal effect estimation
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(17, 3)
    plot_contrast_matrix(trial_contrasts['overall'] , design_matrix=design_matrix.iloc[:,0:40], ax = ax)
    fig.savefig(f'{out_dir}{sub}/figures/{sub}_contrast_overall.png')
    
    fmri_img_smoothed = image.smooth_img(fmri_img, fwhm=6)
    fmri_glm_smoothed = fmri_glm.fit(fmri_img_smoothed, events, confounds=confounds_clean)
    
    efect_of_survey = fmri_glm_smoothed.compute_contrast(np.asarray(trial_contrasts['overall']), output_type='z_score')

    p = plot_stat_map(efect_of_survey, threshold=3,
                  display_mode='z', cut_coords=8, black_bg=False,
                  title='Overall')
    p.savefig(f'{out_dir}{sub}/figures/{sub}_Q_overall_zmap.png')
    nib.save(efect_of_survey, f'{out_dir}{sub}/{sub}_{task}_Q_overal_zmap.nii.gz')
    print(f'Done')
    print('')