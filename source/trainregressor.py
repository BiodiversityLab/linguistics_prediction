from utils.load_dataset import load_data
from utils.load_json_configuration import LoadJSONConfiguration
from utils.module_loader import load_module
from utils.metadata_writer import write_dataset_metadata, write_run_parameters, get_feature_names
from utils.plot import plot_loss, plot_mc_dropout, plot_feature_importance, plot_latitude_error
from utils.default_functions import default_to_device, default_cat, default_permutation_importance
import argparse
import os
from torch import nn
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import mlflow
import numpy as np
import torch
from shapely.geometry import Polygon
import geopandas as gpd

class RegressorDataLoader:
    def __init(self,dataset):
        self.dataset = dataset
   


    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return self.dataset.images[x] ,self.dataset.labels[x]

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

import pdb

def main(args):

    dir_name = create_run_dir(os.path.join('run',args.run_path))
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exclude_channels = [int(channel) for channel in args.permutation_feature_importance_exclude_channels]

    #load the configuration
    configuration = LoadJSONConfiguration(os.path.join('architectures', args.architecture, 'configuration.json'), args.configuration)
    dataset_module = load_module('architectures.'+args.architecture+'.'+args.dataset_module)

    to_device_method = lambda X, y, device: default_to_device(X, y, device)
    if configuration.get('custom_to_device_method') is not None:
        to_device_method = dataset_module.to_device
    catenate_method = lambda x: default_cat(x)
    if configuration.get('custom_catenate_method') is not None:
        catenate_method = dataset_module.catenate
    permutation_importance_method = lambda model, X, y, configuration, feature_names, exclude_channels: default_permutation_importance(model, X, y, configuration, feature_names, exclude_channels=[])
    if configuration.get('custom_permutation_importance_method') is not None:
        permutation_importance_method = dataset_module.permutation_importance_method

    #prepare the dataset
    features, labels, target_df, maximum_n_taxa = load_data(configuration, args.feature_data_dir, multi_label=False)
    write_dataset_metadata(features, dir_name, configuration)
    feature_names = get_feature_names(features, configuration)
    write_run_parameters(args, dir_name)

    
    train_dataset, val_dataset, test_dataset = dataset_module.prepare_dataset(features, labels, target_df, configuration)
    train_dataloader = DataLoader(train_dataset)
    val_dataloader = DataLoader(val_dataset)
    test_dataloader = DataLoader(test_dataset)

    #load the model
    model_module = load_module('architectures.'+args.architecture+'.' + args.model)
    input_channels = len(feature_names)

    model = model_module.load_model(input_channels, configuration)
    model = model.to(device)
    if args.weights != '':
        model.load_state_dict(torch.load(args.weights ,map_location=device))

    l1loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    
    epochs = args.epochs
    best_loss = float('inf')
    train_losses = []
    val_losses = []
  
    # Train the model
    start = time.time()

    train_loss = train(model, train_dataset, val_dataset, test_dataset)
    train_losses.append(train_loss)

    val_MSEloss, val_MAEloss = test(val_dataloader, model, l1loss_fn, loss_fn, device, configuration, feature_names, exclude_channels, dir_name, to_device_method, catenate_method, permutation_importance_method, maximum_n_taxa)
    mlflow.log_metric("val_MSEloss", val_MSEloss)
    mlflow.log_metric("val_MAEloss", val_MAEloss)

    stop = time.time()
    print(f"Done in {stop - start:0.2f} seconds!")
    test_MSEloss, test_MAEloss = test(test_dataloader, model, l1loss_fn, loss_fn, device, configuration, feature_names, exclude_channels, dir_name, to_device_method, catenate_method, permutation_importance_method, maximum_n_taxa, final_epoch=True)
    mlflow.log_metric("test_MSEloss", test_MSEloss)
    mlflow.log_metric("test_MAEloss", test_MAEloss)
    mlflow.log_metric("validation length", len(val_dataloader))
    mlflow.log_metric("test length", len(test_dataloader))

    plot_loss(train_losses, save_path=dir_name)

    for filename in os.listdir(dir_name):
        if filename.endswith('.png'):
            file_path = os.path.join(dir_name, filename)
            mlflow.log_artifact(file_path)

    #predict tiles:
    prediction_polygons = gpd.GeoDataFrame({'geometry': [], 'predictions': []})
    grid_cell_size = 25000
    for dataloader in [train_dataloader, val_dataloader, test_dataloader]:
        for X, y, coordinates  in dataloader:
            X, y = to_device_method(X,y, device)
           # pdb.set_trace()
            pred = model.predict(X)
            
            for index_coord, coord in enumerate(coordinates):
                x0 = int(coord.split('_')[1].split('.')[0])-(grid_cell_size/2)
                y0 = int(coord.split('_')[0].split('.')[0])-(grid_cell_size/2)
                x1 = x0 + grid_cell_size
                y1 = y0 + grid_cell_size
                polygon = Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
                #pdb.set_trace()
                prediction_polygons = prediction_polygons.append({'geometry': polygon, 'predictions': pred[index_coord]*maximum_n_taxa}, ignore_index=True)
    prediction_polygons.crs = "ESRI:54034"
    prediction_polygons.to_file(os.path.join(dir_name, 'predictions.gpkg'), driver="GPKG")


def create_run_dir(run_basepath: Path):
    index = 0
    dir_name = run_basepath

    while os.path.exists(dir_name):
        index += 1
        dir_name = f"{run_basepath}_{index}"

    os.makedirs(dir_name)
    return dir_name


def train(model, train_dataset, val_dataset, test_dataset):
 
    X, y = train_dataset.images,train_dataset.labels
   # Assuming 'X' and 'y' are PyTorch tensors
    X_array = X.numpy()
    y_array = y.numpy().ravel()
    #pdb.set_trace()
    model.fit(X_array, y_array)
    return


def test(dataloader, model, maeloss_fn, mse_fn, device, configuration, feature_names, exclude_channels, experiment_path, to_device_method, catenate_method, permutation_importance_method, maximum_n_taxa, final_epoch=False):
  #  model.eval()
    val_MSEloss = 0
    mae_loss = 0
    X_samples = []
    y_samples = []

    with torch.no_grad():
        for X, y, coords in dataloader:
            X, y = to_device_method(X,y, device)
            
            X_samples.append(X)
            y_samples.append(y)

            # Compute prediction error
            pred = model.predict(X)

            pred = torch.tensor(pred)
            val_MSEloss += mse_fn(pred, y).item()
            mae_loss += maeloss_fn(pred, y).item()

    val_MSEloss /= len(dataloader)
    mae_loss /= len(dataloader)
    mae_loss *= maximum_n_taxa
    print(f"MSEloss: {val_MSEloss:>8f} \n")
    print(f"Average rescaled l1 valid loss: {mae_loss:>8f} \n")
   # pdb.set_trace()
    if final_epoch:
        mc_dropout_mean, mc_dropout_std, mc_labels = predict_mc_dropout(dataloader, model, device, num_samples=100, to_device_method=to_device_method, maximum_n_taxa=maximum_n_taxa)
        plot_mc_dropout(mc_labels, mc_dropout_mean, mc_dropout_std, save_path=experiment_path, maximum_n_taxa=maximum_n_taxa)

       # avg_errors, lats = predict_latitude_error(feature_names, dataloader, model, device, num_samples=100, to_device_method=to_device_method, maximum_n_taxa=maximum_n_taxa)   
       # plot_latitude_error( avg_errors, lats, save_path=experiment_path)

        
    return val_MSEloss, mae_loss

def predict_mc_dropout(dataloader, model, device, num_samples, to_device_method, maximum_n_taxa):

    predictions = []
    labels = []
    sample_predictions = []

    for X, y, coords in dataloader:
        X, y = to_device_method(X,y, device)

        pred = model.predict(X)
        pred = torch.tensor(pred)
        sample_predictions.extend(pred.cpu())
        predictions.append(sample_predictions)



    for X, y, coords in dataloader:
        labels.extend(y.detach().cpu().numpy())
    avg_predictions = np.mean(np.stack(predictions), axis=0)
    std_predictions = np.std(np.stack(predictions), axis=0)
    avg_predictions *= maximum_n_taxa
    std_predictions *= maximum_n_taxa

    labels = np.array(labels)
    labels *= maximum_n_taxa
    return avg_predictions, std_predictions, labels


def get_lat_values(X, index_of_lat):
    lats = []
    for sample in X:
     
        lats.append(sample[index_of_lat].cpu())
    return lats

    
def predict_latitude_error(feature_names, dataloader, model, device, num_samples, to_device_method, maximum_n_taxa):
    '''
    Currently only working with dataset that has format:
    X['constants'] == tensor array with single values
    X == tensor array with single values
    
    '''
    #model.train()  # Set model to train mode
    predictions = []
    errors = []
    labels = []
    lats = []
    index_of_lat = feature_names.index('lat')
    sample_predictions = []
    sample_errors = []
    lat = []
    for X, y, coords in dataloader:
        X, y = to_device_method(X,y, device)
    
        pred = model.predict(X)

        pred = torch.tensor(pred)
        error = abs(pred-y.squeeze())
        sample_errors.extend(error.cpu())
        sample_predictions.extend(pred.cpu())
        predictions.append(sample_predictions)
        errors.append(sample_errors)

    for X, y in dataloader:
        if type(X)==dict:
            lats.extend(get_lat_values(X['constants'], index_of_lat))
        else:
            lats.extend(get_lat_values(X, index_of_lat))
        labels.extend(y.cpu())
    avg_errors = np.mean(np.stack(errors), axis=0)

    avg_errors *= maximum_n_taxa
    return avg_errors, np.array(lats)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture', action='store', type=str, help='name of the model architecture to use')
    parser.add_argument('--model', action='store', type=str, help='name of the model class to use')
    parser.add_argument('--configuration',  action='store', type=str, help='the model configuration to be used')
    parser.add_argument('--dataset_module',  action='store', type=str, help='the dataset to be used')
    parser.add_argument('--epochs', action='store', type=int, help='number of epochs')
    parser.add_argument('--run_path', action='store', type=str, default='train', help='path to store the experiment')
    parser.add_argument('--mlflow_experiment', action='store', default='default')
    parser.add_argument('--permutation_feature_importance_exclude_channels', nargs='+', type=int, default=[], help='Set channels to not include in feature importance test')
    parser.add_argument('--weights', action='store', type=str, default='')
    parser.add_argument('--feature_data_dir', action='store', type=str, default='features')
    parser.add_argument('--device', action='store', type=str, default='0')
    args = parser.parse_args()
    
    mlflow.set_experiment(args.mlflow_experiment)
    args_dict = vars(args)

   


    with mlflow.start_run():
        for key, value in args_dict.items():
            mlflow.log_param(key, value)
        mlflow.log_artifact(os.path.join('architectures', args.architecture, 'configuration.json'))
        mlflow.log_artifact(os.path.join('architectures',args.architecture,args.dataset_module+'.py'))
        mlflow.log_artifact(os.path.join('architectures',args.architecture, args.model+'.py'))
        main(args)
        mlflow.end_run()
