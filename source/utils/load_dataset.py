import os
import pandas as pd
import torch
import numpy as np
import tqdm
import pdb
def get_tif_files_in_directory(parent_dir):
    tif_file_list = []

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".tif"):

                tif_file_list.append([root, file])
    return tif_file_list

def load_inference_data(configuration: dict, feature_data_dir: str):
    '''
    Loads feature. Retrieves the paths to the chosen features from the configuration 

    Args:
        configuration (dict)
        feature_data_dir (feature data folder name)
    
    Returns:
        selected_images (list of image paths)

    '''
   

    feature_dir = f'data/{feature_data_dir}/'
    tifs = get_tif_files_in_directory(feature_dir)
    images = []
    selected_images = []
    samples = np.unique([coords[1].split('[')[-1].split(']')[0] for coords in tifs])
    for index, sample in enumerate(tqdm.tqdm(samples)):

        sample_images = [element for element in tifs if element[1].startswith('['+sample + ']')]
        if 'features' in configuration and len(configuration['features']) > 0:
            
            #get only selected from configuration
            sample = [element for element in sample_images if any(']'+feature+'.' in element[1] for feature in configuration['features'])]
            if len(sample)== len(configuration['features']):
                selected_images.append(sample)
           
   # [element for element in tifs if element[1].startswith('['+sample + ']')]#
    return selected_images

def load_data(configuration: dict, feature_data_dir: str, multi_label: bool):
    '''
    Loads feature and labels paths into correlated lists. Retrieves the paths to the chosen features from the configuration 

    Args:
        configuration (dict)
        feature_data_dir (feature data folder name)
    
    Returns:
        sample_ids (dict)
        labels (torch.Tensor)
        target_df (pandas.core.frame.DataFrame)

    '''
    proj_dir = ""

    #read other dataset
    target_df = pd.read_csv(proj_dir+'data/target/overlapped.csv')

    # Normalize the label data
    maximum_n_overlapped = target_df['overlapped'].max()
    maximum_n_quadratic_overlapped = target_df['quadratic_overlapped'].max()
    #target_df['n_taxa_rescaled'] = target_df['n_taxa'] / target_df['n_taxa'].max()
    target_df['overlapped_rescaled'] = target_df['overlapped'] / maximum_n_overlapped
    target_df['quadratic_overlapped_rescaled'] = target_df['quadratic_overlapped'] / maximum_n_quadratic_overlapped

   
   # labels = torch.tensor(target_df['overlapped_rescaled'], dtype=torch.float32).unsqueeze(1)
    labels = []
    # TESTING
    # small subset
    target_df = target_df[:]
    ####

    # Import the feature data
    feature_dir = proj_dir+f'data/{feature_data_dir}/'
 
    # list of all tif images in feature dir
    tifs = get_tif_files_in_directory(feature_dir)
    
    # create a dicts with key as unique_id and values as paths to features
    sample_ids = {key: None for key in target_df['ID']}
    selected_sample_ids = {key: None for key in target_df['ID']}
    sorted_reduced_sample_ids = {key: None for key in target_df['ID']}

    
    for sample in sample_ids:
        #get all samples
       
        sample_ids[sample] = [element for element in tifs if element[1].startswith('['+sample + ']')]# or element[1].startswith(sample[:7] + 'all')]
                              #or element[1].startswith(sample.split('_')[0]) 
                             # and element[1].split('_')[2].startswith('monthly')
                             # and target_df[target_df['unique_id']==sample]['month'].item() == element[1].split('_')[1]]
  
        if 'features' in configuration and len(configuration['features']) > 0:
            #get only selected from configuration
            selected_sample_ids[sample] = [element for element in sample_ids[sample] if any(']'+feature+'.' in element[1] for feature in configuration['features'])]
            if multi_label:
                labels.append([target_df.loc[target_df['ID']==sample]['overlapped_rescaled'].iloc[0],target_df.loc[target_df['ID']==sample]['quadratic_overlapped_rescaled'].iloc[0]])
            else:
                labels.append(target_df.loc[target_df['ID']==sample]['overlapped_rescaled'].iloc[0])
                
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)    
    if 'features' in configuration and len(configuration['features']) > 0:
        def find_feature_index(element):
            for index, feature in enumerate(configuration['features']):
                if '_'+feature+'.' in element[1]:
                    return index
            return len(configuration['features'])
        #sort the order of channels/features, making sure it is the same order as defined in configuration.json
        for sample, elements in selected_sample_ids.items():
            sorted_elements = sorted(elements, key=find_feature_index)
            sorted_reduced_sample_ids[sample] = sorted_elements

        return sorted_reduced_sample_ids, labels, target_df, maximum_n_overlapped

    return sample_ids, labels, target_df, maximum_n_overlapped


def select_train_val_test(df, val_fraction=0.2, test_fraction=0.2, shuffle=True, seed=None):
    all_indices = np.arange(len(df))
   
    trap_ids = df['ID'].unique()

    if shuffle:
        if not seed:
            seed = np.random.randint(0, 999999999)
        np.random.seed(seed)
        print('Shuffling data, using seed', seed)

        # Shuffling based in trap ID to separate traps in the train, val and test datasets
        shuffled_trap_ids = np.random.choice(trap_ids, len(trap_ids), replace=False)
        shuffled_indices = []
        for i in range(len(shuffled_trap_ids)):
            ids = df[df['ID'] == shuffled_trap_ids[i]].index.tolist()
            shuffled_indices.extend(ids)
    else:
        shuffled_indices = all_indices

    # select train, validation, and test data
    n_test_instances = np.round(len(shuffled_indices) * test_fraction).astype(int)
    n_val_instances = np.round(len(shuffled_indices) * val_fraction).astype(int)
    test_idx = shuffled_indices[:n_test_instances]
    val_idx = shuffled_indices[n_test_instances:n_test_instances + n_val_instances]
    train_idx = shuffled_indices[n_test_instances + n_val_instances:]

    return train_idx, val_idx, test_idx