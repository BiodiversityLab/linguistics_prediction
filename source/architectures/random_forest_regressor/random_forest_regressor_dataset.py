import os
import imageio.v2 as imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from utils.load_dataset import select_train_val_test
import pdb
from tqdm import tqdm
import numpy as np

class SingleValueDataset(Dataset):
    def __init__(self, features, labels, transform=None, configuration=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.images = []
        self.coordinates = []
        for idx, image_paths in tqdm(enumerate(self.features)):
            self.images.append([])
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

                self.images[-1].extend([image.min(), image.max(), image.mean(), np.median(image), image.std()])
            self.coordinates.append(image_paths[0][1].split('[')[-1].split(']')[0])
        self.images = np.array(self.images)

        self.images = torch.tensor(self.images, dtype=torch.float)
        


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_tensor = self.images[idx]
        return image_tensor, label, self.coordinates[idx]


def prepare_dataset(features, labels, target_df, configuration: dict):
    
    # Split the features into train, val and test datasets
    train_ids, val_ids, test_ids = select_train_val_test(target_df, seed=7)

    train_features = [features[sample] for sample in target_df.iloc[train_ids]['ID']]
    train_labels = labels[train_ids]
    val_features = [features[sample] for sample in target_df.iloc[val_ids]['ID']]
    val_labels = labels[val_ids]
    test_features = [features[sample] for sample in target_df.iloc[test_ids]['ID']]
    test_labels = labels[test_ids]

    transform = transforms.Compose([transforms.ToTensor()])
    # Make the datasets PyTorch compatible
    train_dataset = SingleValueDataset(train_features, train_labels, transform=transform, configuration=configuration)
    val_dataset = SingleValueDataset(val_features, val_labels,transform=transform, configuration=configuration)
    test_dataset = SingleValueDataset(test_features, test_labels, transform=transform, configuration=configuration)

    return train_dataset, val_dataset, test_dataset
