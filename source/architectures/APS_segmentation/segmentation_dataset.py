import os
import imageio.v2 as imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from utils.load_dataset import select_train_val_test
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import pdb
try:
    import gdal
except:
    from osgeo import gdal
    import osgeo.osr as osr


def reshape_image(image):
    if image.shape[0] != image.shape[1]:
        
        diff = np.abs(image.shape[0] - image.shape[1])
        if image.shape[0] > image.shape[1]:
            # Lägg till en extra kolumn av padding på höger sida
            if len(image.shape) > 2:
                pad_width = ((0, 0), (0, diff), (0,0))
            else:
                pad_width = ((0, 0), (0, diff))
        else:
            if len(image.shape) > 2:
                # Lägg till en extra rad av padding längs nedre kanten
                pad_width = ((0, diff), (0, 0), (0,0))
            else:
                    pad_width = ((0, diff), (0, 0))
        image = np.pad(image, pad_width, mode='constant')
    
    if image.shape[0] % 16 != 0:
        pad = 16 - image.shape[0] % 16
        if len(image.shape) > 2:
            pad_width = ((0, pad), (0, pad), (0,0))
        else:
            pad_width = ((0, pad), (0, pad))
        image = np.pad(image, pad_width, mode='constant')
    return image


class SegmentationDataset(Dataset):
    def __init__(self, features, labels, transform=None, configuration=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.images = []
       # self.constants = []
        self.coordinates = []
        self.images_annotations = []
        with open('filtered_list.txt', 'r') as file: 
            lines = [line.strip() for line in file]
        #pdb.set_trace()
        for idx, image_paths in tqdm(enumerate(self.features)):
           # pdb.set_trace()
            if len(image_paths) == 0:
                continue
            img_batch = []
            #const_batch = []
            
            if image_paths[0][1].split('[')[1].split(']')[0] not in lines:
                continue
            for feature_index, path in enumerate(image_paths):
                
                image = imageio.imread(os.path.join(path[0], path[1]))
                data_source_name = path[0].split('/')[-1]
                if data_source_name in configuration['channels'] :
                    channels_to_keep = configuration['channels'][data_source_name]
                    new_image = []
                    for channel in channels_to_keep:
                        new_image.append(image[:,:,channel])
                    image = np.array(new_image).transpose(1,2,0)
                image[image < 0] = 0

                image = reshape_image(image)

                if self.transform:
                    image = self.transform(image)
                if len(image.shape) > 2:
                    for channel in image:
                        img_batch.append(torch.tensor(channel.squeeze(0), dtype=torch.float))
                else:
                    img_batch.append(torch.tensor(image.squeeze(0), dtype=torch.float))

            self.coordinates.append(image_paths[0][1].split('[')[-1].split(']')[0])
            
           
            self.images.append(torch.stack(img_batch))
            #self.constants.append(torch.tensor(const_batch, dtype=torch.float))

       

        #feature normalization
        features_mean = [0]*len(self.images[0])
        features_std = [0]*len(self.images[0])
        for sample in self.images:
            for feature_index, feature in enumerate(sample):
                features_mean[feature_index] += feature.mean()
                features_std[feature_index] += feature.std()

        for sample in self.images:
            for feature_index, feature in enumerate(sample):
                feature = (feature-features_mean[feature_index])/features_std[feature_index]


        for coord in self.coordinates: 
            ds = gdal.Open(os.path.join('../../annotations/',coord+'.tif'))
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            annotation_image = self.transform(arr)
            if len(annotation_image.shape) > 2:
                annotation_image = annotation_image.squeeze(0)
            annotation_image = reshape_image(annotation_image)
            annotation_image /= 255.0
            self.images_annotations.append(torch.tensor([annotation_image]))
       # pdb.set_trace()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.images_annotations[idx], self.coordinates[idx]

def permutation_importance_method(model, X, y, configuration, feature_names, metric=mean_squared_error, exclude_channels=[]):
    model.eval()
    #feature_names = configuration['features']
    if len(y.shape) > 2:
        baseline_performance = metric(y[:,0,:].cpu().detach().numpy(), model(X).detach().numpy())
    else:
        baseline_performance = metric(y.cpu().detach().numpy(), model(X).cpu().detach().numpy())
    importance_scores = []
    

    adjusted_importance_scores = [score for i, score in enumerate(importance_scores) if i not in exclude_channels]
    adjusted_feature_names = [name for i, name in enumerate(feature_names) if i not in exclude_channels]

    for i in range(X.shape[1]): #for loop in feature dimension
            
        X_permuted = X.clone().detach()
        X_permuted[:, i] = shuffle(X_permuted[:, i])
        if len(y.shape) > 2:
            permuted_performance = metric(y[:,0,:].cpu().detach().numpy(), model(X_permuted).cpu().detach().numpy())
        else:
            permuted_performance = metric(y.cpu().detach().numpy(), model(X_permuted).cpu().detach().numpy())

        importance = abs(baseline_performance - permuted_performance)
        importance_scores.append(importance)
    return importance_scores, adjusted_feature_names

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
 
    train_dataset = SegmentationDataset(train_features, train_labels, transform=transform, configuration=configuration)
    val_dataset = SegmentationDataset(val_features, val_labels,transform=transform, configuration=configuration)
    test_dataset = SegmentationDataset(test_features, test_labels, transform=transform, configuration=configuration)

    return train_dataset, val_dataset, test_dataset



def prepare_dataset_cross_validation(features, labels, target_df, configuration: dict, dataset_index: int, n_splits=5, shuffle=True, seed=1):
    # Skapa en KFold instans med det önskade antalet splits, shuffle och seed.
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    # Förberedd transformering för PyTorch-dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Blanda alla unika ID:n en gång
    all_ids = target_df['ID'].unique()
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_ids)

    # Dela upp ID:n i 5 lika stora delar
    id_chunks = np.array_split(all_ids, n_splits)

  #  for i in range(n_splits):
    # Välj en del för test, en annan för val och resten för träning
    test_ids = id_chunks[dataset_index]
    val_ids = id_chunks[(dataset_index + 1) % n_splits]
    train_ids = np.concatenate([chunk for j, chunk in enumerate(id_chunks) if j != dataset_index and j != (dataset_index + 1) % n_splits])

    # Konvertera ID:n till index i hela dataframe
    train_indices = target_df[target_df['ID'].isin(train_ids)].index
    val_indices = target_df[target_df['ID'].isin(val_ids)].index
    test_indices = target_df[target_df['ID'].isin(test_ids)].index

    # Skapa dataset för träning, validering och test
    #pdb.set_trace()
    train_features = [features[sample] for sample in train_ids]
    train_labels = labels[train_indices]

    val_features = [features[sample] for sample in val_ids]
    val_labels = labels[val_indices]

    test_features = [features[sample] for sample in test_ids]
    test_labels = labels[test_indices]

    # Skapa PyTorch-datasets
    train_dataset = SegmentationDataset(train_features, train_labels, transform=transform, configuration=configuration)
    val_dataset = SegmentationDataset(val_features, val_labels, transform=transform, configuration=configuration)
    test_dataset = SegmentationDataset(test_features, test_labels, transform=transform, configuration=configuration)

        # Lägg till datan för denna fold till listan
        #dataset_splits.append((train_dataset, val_dataset, test_dataset))

    return train_dataset, val_dataset, test_dataset