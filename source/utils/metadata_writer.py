import json
import imageio.v2 as imageio
import os
from pathlib import Path
from tifffile import imread
import pdb
def get_feature_names(features: list, configuration: dict):
    all_images = []   
    first_sample_id = next(iter(features))

    for feature in features[first_sample_id]:
            reduced_feature_name = feature[1]
            reduced_feature_name = reduced_feature_name.split(']')[1].split('.')[0]
            #reduced_feature_name = reduced_feature_name.split('_')[-1]
          
            img = imread(os.path.join(feature[0],feature[1]))
           # pdb.set_trace()
            data_source_name = feature[0].split('/')[-1]
            if 'channels' in configuration and data_source_name in configuration['channels']:
                    channels_to_keep = configuration['channels'][data_source_name]
                    new_image = []
                    for channel_index, channel in enumerate(channels_to_keep):
                        if 'channels_name' in configuration and data_source_name in configuration['channels_name']:
                             all_images.append(reduced_feature_name+f'_{configuration["channels_name"][data_source_name][channel_index]}')
                        else:
                             all_images.append(reduced_feature_name+f'_{channel}')
                       # new_image.append(image[:,:,channel])
                    #image = np.array(new_image).transpose(1,2,0)

            elif len(img.shape) > 2:
                
                number_of_channels = img.shape[2]
                for chan in range(number_of_channels):
                    all_images.append(reduced_feature_name+f'_{chan}')
            else:
                all_images.append(reduced_feature_name)
    unique_names = []
    for feature in all_images:
        if feature not in unique_names:
            unique_names.append(feature)

    return unique_names


def get_feature_names_test(features: list, configuration: dict):
    all_images = []   
    for feature in features[0]:
            reduced_feature_name = feature[1]
        #    pdb.set_trace()
            reduced_feature_name = reduced_feature_name.split(']')[1].split('.')[0]
            #reduced_feature_name = reduced_feature_name.split('_')[-1]
          
            img = imread(os.path.join(feature[0],feature[1]))
           # pdb.set_trace()
            data_source_name = feature[0].split('/')[-1]
            if 'channels' in configuration and data_source_name in configuration['channels']:
                    channels_to_keep = configuration['channels'][data_source_name]
                    new_image = []
                    for channel_index, channel in enumerate(channels_to_keep):
                        if 'channels_name' in configuration and data_source_name in configuration['channels_name']:
                             all_images.append(reduced_feature_name+f'_{configuration["channels_name"][data_source_name][channel_index]}')
                        else:
                             all_images.append(reduced_feature_name+f'_{channel}')
                       # new_image.append(image[:,:,channel])
                    #image = np.array(new_image).transpose(1,2,0)

            elif len(img.shape) > 2:
                
                number_of_channels = img.shape[2]
                for chan in range(number_of_channels):
                    all_images.append(reduced_feature_name+f'_{chan}')
            else:
                all_images.append(reduced_feature_name)
    unique_names = []
    for feature in all_images:
        if feature not in unique_names:
            unique_names.append(feature)

    return unique_names

def write_dataset_metadata(features: list, run_path: Path, configuration: dict):

    first_sample_id = next(iter(features))
    unique_names = get_feature_names(features, configuration)
    

    metadata = {}
   # pdb.set_trace()
    #read image and count channels
    for img, name in zip(features[first_sample_id],unique_names):

        img_array = imageio.imread(os.path.join(img[0],img[1]))
        if len(img_array.shape) == 2:
            metadata[name] = 1
        else:
          
            #len(img_array.shape[2])
            metadata[name] = img_array.shape[2]

    with open(os.path.join(run_path, 'metadata.json'), 'w', encoding='utf-8') as f:
         json.dump(metadata, f, ensure_ascii=False, indent=4)
    return unique_names

def write_run_parameters(args, run_path: Path):
     
    args_dict = vars(args)
    with open(os.path.join(run_path,'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)