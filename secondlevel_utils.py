from collections import OrderedDict as odict
from functools import partial
from glob import glob
from joblib import Parallel, delayed
import numpy as np
from os import makedirs, path, sep
import pandas as pd
import pickle
import re
import shutil

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

#fmri imports
import nibabel
from nilearn import datasets, image, input_data, masking
from nilearn.decomposition import CanICA
from nipype.caching import Memory
from nipype.interfaces import fsl


# ********************************************************
# Misc Functions
# ********************************************************
def get_contrast_names(subjectinfo_path):
    try:
        contrasts = pickle.load(open(subjectinfo_path, 'rb')).contrasts
        contrast_names = [i[0] for i in contrasts]
    except IndexError:
        print('No subjectinfo found for %s_%s' % (task, model))
        contrast_names = None
    return contrast_names

def create_group_mask(mask_loc, fmriprep_dir, threshold=.8, verbose=True):
    if verbose:
        print('Creating Group mask...')
    makedirs(path.dirname(mask_loc), exist_ok=True)
    brainmasks = glob(path.join(fmriprep_dir,'sub-s???',
                               '*','func','*MNI152NLin2009cAsym_brainmask*'))
    mean_mask = image.mean_img(brainmasks)
    group_mask = image.math_img("a>=%s" % str(threshold), a=mean_mask)
    group_mask.to_filename(mask_loc)
    if verbose:
        print('Finished creating group mask')
        
# ********************************************************
# Functions to create Group Maps
# ********************************************************
def save_tmaps(copes_loc,
               mask_loc,
               working_dir,
               permutations,
               rerun=False):
    task_dir = path.dirname(copes_loc)
    contrast_name = path.basename(copes_loc).split('_cope')[0].rstrip('.nii.gz')
    contrast_working_dir = path.join(working_dir, path.basename(copes_loc))
    tfile_loc = path.join(task_dir, "%s_raw_tfile.nii.gz" % contrast_name)
    tfile_corrected_loc = path.join(task_dir,
                               "%s_corrected_tfile.nii.gz" % contrast_name)
    makedirs(contrast_working_dir, exist_ok=True)
    # perform permutation test to assess significance
    if not path.exists(tfile_loc) or rerun:
        mem = Memory(base_dir=contrast_working_dir)
        randomise = mem.cache(fsl.Randomise)
        randomise_results = randomise(
            in_file=copes_loc,
            mask=mask_loc,
            one_sample_group_mean=True,
            tfce=True,  # look at paper
            vox_p_values=True,
            var_smooth=10,
            num_perm=permutations)
        # save results
        raw_tfile = randomise_results.outputs.tstat_files[0]
        corrected_tfile = randomise_results.outputs.t_corrected_p_files[0]
        shutil.move(raw_tfile, tfile_loc)
        shutil.move(corrected_tfile, tfile_corrected_loc)
        shutil.rmtree(contrast_working_dir)
    return tfile_loc, tfile_corrected_loc

# ********************************************************
# Functions to get fmri maps and get/create parcellations
# ********************************************************
def get_map_files(first_level_dir,
                  tasks,
                  model,
                  map_prefix='zstat',
                  selectors='default'):
    map_files = odict()
    # select only a subset of contrasts (i.e.  get explicit contrasts, not vs rest)
    if selectors == 'default':
        selectors = ['-', 'response_time',
                    'network',
                    'EV', 'risk', #CCT
                    'subjective_value', 'LL_vs_SS', #discount
                     'cue_switch', 'task_switch', #twoByTwo
                    'search_depth'] #WATT3
    elif selectors is None:
        selectors = []
    for task in tasks: 
        subjectinfo_paths = glob(path.join(first_level_dir,'*', task, model, 'wf-contrast', 'subjectinfo.pkl'))
        if len(subjectinfo_paths) > 0:
            contrast_names = get_contrast_names(subjectinfo_paths[0])
        else:
            print("No subjectinfo found for %s, Model-%s" % (task, model))
            continue
        for i, name in enumerate(contrast_names):
            if any([sel in name for sel in selectors]) or len(selectors)==0:
                map_files[task+'_'+name] = sorted(glob(path.join(first_level_dir,
                                                            '*', 
                                                             task,
                                                             model,
                                                             'wf-contrast', 
                                                             '%s%s.nii.gz' % (map_prefix, str(i+1)))))
    return map_files

def get_group_maps(second_level_dir,
                    tasks,
                    model,
                    match_string='copes'):
    map_files = odict()
    for task in tasks: 
        group_files = sorted(glob(path.join(second_level_dir, task, 
                                             model, 'wf-contrast','*%s*.nii.gz' % match_string)))
        for filey in group_files:
            contrast_name = path.basename(filey).split(match_string)[0].rstrip('_')
            img = image.load_img(filey)
            if len(img.shape)==4:
                img = image.mean_img(img)
            map_files[task+'_'+contrast_name] = img
    return map_files

def flatten(lst):
    # flattens list of lists
    return [item for sublist in lst for item in sublist]

def get_metadata(map_files):
    tasks = []
    contrast_names = []
    for k,v in map_files.items():
        try:
            tasks += [k.split('_')[0]]*len(v)
            contrast_names += ['_'.join(k.split('_')[1:])]*len(v)
        except TypeError:
            tasks += [k.split('_')[0]]
            contrast_names += ['_'.join(k.split('_')[1:])]
    if len(tasks) != len(map_files): # thus there must have been multiple images per contrast
        out = flatten(map_files.values())
        subjects = [i.split(sep)[-5] for i in out]
        df = pd.DataFrame({'task': tasks,
                         'contrast_name': contrast_names,
                         'subject': subjects})
    else:
        df = pd.DataFrame({'task': tasks,
                         'contrast_name': contrast_names})
    return df

def concat_map_files(map_files, file_type,second_level_dir, model, verbose=False, rerun=True):
    filenames = []
    for k,v in map_files.items():
        if verbose: print("Concatenating %s" % k)
        task, *contrast = k.split('_')
        contrast_name = '_'.join(contrast)
        filename = path.join(second_level_dir, task, model,
                                          'wf-contrast', 'task-%s_contrast-%s_file-%s_concat.nii.gz' % (task, contrast_name, file_type))
        if rerun or not(path.exists(filename)):
            makedirs(path.dirname(filename), exist_ok=True)
            concat_image = image.concat_imgs(v)
            concat_image.to_filename(filename)
        filenames.append(filename)
    return filenames

def smooth_concat_files(concat_files, fwhm=4.4, verbose=False, rerun=True):
    filenames = []
    for filey in concat_files:
        if verbose: print("Smoothing %s" % k)
        smoothed = image.smooth_img(filey, fwhm)
        smooth_name = filey.replace('concat', 'concatsmoothed-fwhm%s' % str(fwhm))
        if rerun or not(path.exists(smooth_name)):
            smoothed.to_filename(smooth_name)
        filenames.append(smooth_name)
    return filenames

def get_mean_maps(image_list, contrast_name_list, save=True, rerun=True):
    assert len(image_list) == len(contrast_name_list)
    map_files = odict()
    for contrast_name,filey in zip(contrast_name_list, image_list):
        filename = path.join(path.dirname(filey),path.basename(filey).rstrip('nii.gz')+'_mean.nii.gz')
        if not path.exists(filename) or rerun:
            d = path.dirname(filey)
            name = path.basename(filey)
            mean_img = image.mean_img(image.load_img(filey))
            if save:
                mean_img.to_filename(filename)
        else:
            mean_img = image.load_img(filename)
        map_files[contrast_name] = mean_img
    return map_files
    
def get_ICA_parcellation(map_files,
                         mask_loc,
                         working_dir,
                         second_level_dir,
                         n_comps=20,
                         smoothing=4.4,
                         filename=None):
    try:
        map_files = flatten(map_files.values())
    except AttributeError:
        pass
    group_mask = nibabel.load(mask_loc)
    ##  get components
    canica = CanICA(mask = group_mask, n_components=n_comps, 
                    smoothing_fwhm=smoothing, memory=path.join(working_dir, "nilearn_cache"), 
                    memory_level=2, threshold=3., 
                    verbose=10, random_state=0) # multi-level components modeling across subjects
    canica.fit(map_files)
    masker = canica.masker_
    components_img = masker.inverse_transform(canica.components_)
    if filename is not None:
        prefix = filename+'_'
        components_img.to_filename(path.join(second_level_dir, 
                                             'parcellation',
                                            '%scanica%s.nii.gz' 
                                            % (prefix, n_comps)))
    return components_img
    
def get_established_parcellation(parcellation="Harvard_Oxford", target_img=None,
                                parcellation_dir=None):
    if parcellation == "Harvard_Oxford":
        name = "Harvard_Oxford_cort-prob-2mm"
        data = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm', data_dir=parcellation_dir)
        parcel = nibabel.load(data['maps'])
        labels = data['labels'][1:] # first label is background
        atlas_threshold = 25
    elif parcellation == "smith":
        name = "smith_rsn70"
        data = datasets.fetch_atlas_smith_2009(data_dir=parcellation_dir)['rsn70']
        parcel = nibabel.load(data)
        labels = range(parcel.shape[-1])
        atlas_threshold = 4
    elif parcellation == "glasser":
        glasser_dir = path.join(parcellation_dir, 'glasser')
        data = image.load_img(path.join(glasser_dir, 'HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz'))
        parcel = image.new_img_like(data, (data.get_fdata()+.01).astype(int))
        labels = list(np.genfromtxt(path.join(glasser_dir, 'HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt'),
                                    dtype=str, usecols=1))
        name = 'glasser'
        atlas_threshold = None
        # split down midline into lateralized ROIs
        data_coords = np.where(parcel.get_data())
        right_coords = *(i[data_coords[0]>parcel.shape[0]//2] for i in data_coords), # tuple comprehension
        parcel.get_data()[right_coords] += len(labels)
        labels = labels + [l.replace('L_', 'R_') for l in labels]
    if target_img:
        parcel = image.resample_to_img(parcel, target_img, interpolation='nearest')
    return parcel, labels, name, atlas_threshold

def parcel_to_atlas(parcel, threshold):
    # convert parcel to atlas by finding maximum values
    # example below is based on Harvard_Oxford's strategy "cort-maxprob-thr25-2mm"
    data = parcel.get_fdata().copy()
    data[data<threshold] = 0
    atlas=image.new_img_like(parcel, np.argmax(data,3))
    return atlas
# ********************************************************
# Functions to extract ROIs from parcellations
# ********************************************************
def get_ROI_from_parcel(parcel, ROI, threshold=0):
    """
    Extracts ROI from parcel
    If 4D probabilistic parcel, a threshould must be defined to determine the cutoff for the ROI.
    If 3d parcel (atlas), threshold is ignored.
    
    Args:
        parcel: 3D or 4D parcel
        ROI: index of ROI. Should be 0 indexed. For 4D, extract the ROI_i slice in the 4th dimension.
                If 3D, find values that are equal to ROI_i+1 (assumes 0 is used to reflect background)
    """
    if len(parcel.shape) == 4:
        # convert a probabilistic parcellation into an ROI mask
        roi_mask = parcel.get_fdata()[:,:,:,ROI]>threshold 
    else:
        roi_mask = parcel.get_fdata() == (ROI+1)
    assert np.sum(roi_mask) != 0, "ROI doesn't exist. Returned empty map"
    roi_mask = image.new_img_like(parcel, roi_mask)
    return roi_mask

def mask_map_files(roi_i, map_files, parcel, extraction_dir, 
                   metadata=None, labels=None, rerun=True,
                   threshold=0, save=False):
    """
    Extracts an ROI from a parcel and masks a set of fmri maps
    Args:
        parcel: a 3D or 4D parcellation to pass to get_ROI_from_parcel
        roi_i: the index of the parcel
        map_files: a list of 3D or 4D fmri maps
        extraction_dir: the location to save the masked maps
        metadata (optional): metadata the same length as the sum of the map files 4th dimension.
            If 3D map files are passed, this is just the length of the list of files.
        labels (optional): list of labels for the parcellation. Will be used to label the output file
        rerun: If False will not run if a previous file is found
        threshold: threshold passed to get_ROI_from_parcel
    """
    if labels is not None:
        key = labels[roi_i]
    else:
        key = roi_i
    file = path.join(extraction_dir, 'contrasts_ROI-%s_extraction.pkl' % key)
    if not path.exists(file) or rerun:
        print("Masking %s" % key)
        mask_img = get_ROI_from_parcel(parcel, roi_i, threshold)
        masked_map = masking.apply_mask(map_files, mask_img=mask_img)
        # fill 0 values with mean of values
        masked_map[masked_map==0] = np.mean(masked_map)
        if metadata is not None:
            masked_map = pd.concat([metadata, pd.DataFrame(masked_map)], axis=1)
        if save:
            masked_map.to_pickle(file)
            return file
        else:
            return masked_map
    else:
        return pickle.load(open(file, 'rb'))
    
def extract_roi_vals(map_files, parcel, extraction_dir, rois=None, threshold=0,
                     metadata=None, labels=None, rerun=True, n_procs=1, save=True):
    """ 
    Mask nifti images using a parcellation
    
    Runs mask_map_files on each ROI of the parcellation. See mask_map_files for argument definitions
    """
    try:
        map_files = flatten(map_files.values())
    except TypeError:
        map_files = list(map_files.values())
    except AttributeError:
        pass
    if rois is None:
        if len(parcel.shape) == 4:
            rois = range(parcel.shape[-1])
        else:
            rois = range(len(np.unique(parcel.get_data().flatten()))-1)
    out = []
    # parallelize
    if n_procs > 1:
        partial_func = partial(mask_map_files, parcel=parcel, map_files=map_files, 
                                 extraction_dir=extraction_dir, metadata=metadata, 
                                 labels=labels, rerun=rerun, threshold=threshold,
                                 save=save)
        out = Parallel(n_jobs=n_procs)(delayed(partial_func)(roi_i) for roi_i in rois)
    else:
        for roi_i in rois:
            out.append(mask_map_files(roi_i, parcel, map_files, extraction_dir, metadata, labels, rerun, threshold,
                                      save=save))
    return out


# ********************************************************
# RDM functions
# ********************************************************
def get_RDMs(ROI_dict):
    # converts ROI dictionary (returned by extract_roi_vals) of contrast X voxel values to RDMs
    RDMs = odict({})
    for key,val in ROI_dict.items():
        if type(val) == pd.core.frame.DataFrame:
            subset_cols = [c for c in val.columns if type(c) != str]
            val = val.loc[:, subset_cols].values
            # no contrast is allowed to be completely constant, RDMs cannot be calculated
            if np.sum(np.std(val,1)<1E-5) > 0:
                RDMs[key] = None
                continue
        corr = 1-np.corrcoef(val)
        RDMs[key] = corr
    return RDMs
        
# ********************************************************
# 2nd level analysis utility functions
# ********************************************************

# function to get TS within labels
def project_contrast(img_files, parcellation, mask_file, resample=True):
    if type(parcellation) == str:
        parcellation = image.load_img(parcellation)
    resampled_images = image.resample_img(img_files, parcellation.affine)
    if len(parcellation.shape) == 3:
        masker = input_data.NiftiLabelsMasker(labels_img=parcellation, 
                                               resampling_target="labels", 
                                               standardize=False,
                                               memory='nilearn_cache', 
                                               memory_level=1)
    elif len(parcellation.shape) == 4:
         masker = input_data.NiftiMapsMasker(maps_img=parcellation, 
                                             mask_img=mask_file,
                                             resampling_target="maps", 
                                             standardize=False,
                                             memory='nilearn_cache',
                                             memory_level=1)
    time_series = masker.fit_transform(resampled_images)
    return time_series, masker

def create_projections_df(parcellation, mask_file, 
                         data_dir, tasks, filename=None):
    
    # project contrasts into lower dimensional space    
    projections = []
    index = []
    for task in tasks:
        task_files = get_map_files(tasks=[task])
        # for each contrast, project into space defined by parcellation file
        for contrast_name, func_files in task_files.items():
            TS, masker = project_contrast(func_files,
                                          parcellation, 
                                          mask_file)
            projections.append(TS)
            index += [re.search('s[0-9][0-9][0-9]',f).group(0)
                        + '_%s' % (contrast_name)
                        for f in func_files]
    projections_df = pd.DataFrame(np.vstack(projections), index)
    
    # split index into column names
    subj = [i[:4] for i in projections_df.index]
    contrast = [i[5:] for i in projections_df.index]
    projections_df.insert(0, 'subj', subj)
    projections_df.insert(0, 'contrast', contrast)
    
    # save
    if filename:
        projections_df.to_json(filename)
    return projections_df

# functions on projections df
def create_neural_feature_mat(projections_df, filename=None):
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert path.exists(projections_df)
        projections_df = pd.read_json(projections_df)
        
    subj = [i[:4] for i in projections_df.index]
    contrast = [i[5:] for i in projections_df.index]
    projections_df.insert(0, 'subj', subj)
    projections_df.insert(0, 'contrast', contrast)
    neural_feature_mat = projections_df.pivot(index='subj', columns='contrast')
    if filename:
        neural_feature_mat.to_json(filename)
    return neural_feature_mat

def projections_corr(projections_df, remove_global=True, grouping=None):
    """ Create a correlation matrix of a projections dataframe
    
    Args:
        projections_df: a projection_df, as create by create_projection_df
        remove_global: if True, subtract the mean contrast
        grouping: "subj" or "contrast". If provided, average over the group
        
    Returns:
        Correlation Matrix
    """
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert path.exists(projections_df)
        projections_df = pd.read_json(projections_df)
    
    if remove_global:
        projections_df.iloc[:,2:] -= projections_df.mean()
    if grouping:
        projections_df = projections_df.groupby(grouping).mean()
    return projections_df.T.corr()

def get_confusion_matrix(projections_df, normalize=True):
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert path.exists(projections_df)
        projections_df = pd.read_json(projections_df)
        
    X = projections_df.iloc[:, 2:]
    y = projections_df.contrast
    clf = LogisticRegressionCV(multi_class='multinomial')
    predict = cross_val_predict(clf, X, y, cv=10)
    cm = confusion_matrix(y, predict)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
    
                                           
