import os
import imageio.v2 as imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from utils.load_dataset import select_train_val_test
import pdb
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

class SingleValueDataset(Dataset):
    def __init__(self, features, labels, transform=None, configuration=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.images = []
        self.mins = []
        self.means = []
        self.stds = []

        for idx, image_paths in tqdm(enumerate(self.features)):
            self.images.append([])
            self.mins.append([])
            self.means.append([])
            self.stds.append([])
            #self.images.append([])
            for path in image_paths:
                image = imageio.imread(os.path.join(path[0],path[1]))
                data_source_name = path[0].split('/')[-1]
                if data_source_name in configuration['channels'] :
                    channels_to_keep = configuration['channels'][data_source_name]
                    new_image = []
                    for channel in channels_to_keep:
                        new_image.append(image[:,:,channel])
                    image = np.array(new_image).transpose(1,2,0)
                # remove NaN values

                image[image < 0] = 0
                if len(image.shape) > 2:
                    number_of_channels = image.shape[2]
                    for chan in range(number_of_channels):
                        self.mins[-1].extend([image[:,:,chan].min()])
                        self.means[-1].extend([image[:,:,chan].mean()])
                        self.stds[-1].extend([image[:,:,chan].std()])
                        self.images[-1].extend([image.min(), image.max(), image.mean(), np.median(image), image.std()])
                else:
                    self.mins[-1].extend([image.min()])
                    self.means[-1].extend([image.mean()])
                    self.stds[-1].extend([image.std()])

                    self.images[-1].extend([image.min(), image.max(), image.mean(), np.median(image), image.std()])

        self.images = np.array(self.images)

        self.images = torch.tensor(self.images, dtype=torch.float)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_tensor = self.images[idx]
        return image_tensor, label

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

    return all_indices

def prepare_dataset(features, labels, target_df, configuration: dict):
    
    # Split the features into train, val and test datasets
    all_indices = select_train_val_test(target_df, seed=7)

    all_features = [features[sample] for sample in target_df.iloc[all_indices]['ID']]
    all_labels = labels[all_indices]
    
    

    transform = transforms.Compose([transforms.ToTensor()])
    # Make the datasets PyTorch compatible
    train_dataset = SingleValueDataset(all_features, all_labels, transform=transform, configuration=configuration)
    
    labels = np.array(train_dataset.labels)
    means = np.array(train_dataset.means)
    stds = np.array(train_dataset.stds)
    
    for index, feature in enumerate(all_features[0]):
        data_source_name = feature[0].split('/')[-1]
        if 'channels_name' in configuration and data_source_name in configuration['channels_name']:
            for channel in configuration['channels_name'][data_source_name]:
                plot(channel, labels, means, index, "mean")
                plot(channel, labels, stds, index, "std")  
             
        else:
            feature_name = feature[1].split(']')[-1].split('.tif')[0]
            plot(feature_name, labels, means, index, "mean")
            plot(feature_name, labels, stds, index, "std")  
    exit(0)
   # val_dataset = SingleValueDataset(all_features, all_labels,transform=transform)
   # test_dataset = SingleValueDataset(all_features, all_labels, transform=transform)

   # return train_dataset, val_dataset, test_dataset


def plot(feature_name, labels, values, index, value_type):
        
        coefficients = np.polyfit(labels.flatten(), values[:,index].flatten(), 1)
        linear_poly = np.poly1d(coefficients)
        x_values = [np.min(labels.flatten()), np.max(labels.flatten())]
        y_values = linear_poly(x_values)
        plt.plot(x_values, y_values)
        plt.scatter(labels,values[:,index], marker='o', color='b', label='Min plot')
        plt.title(feature_name)
        
        plt.xlabel('labels')
        
        plt.ylabel('values of {}'.format(value_type))
        
        plt.legend()
        

        plt.savefig('../plots/{}_{}.png'.format(feature_name, value_type))
        plt.clf()  