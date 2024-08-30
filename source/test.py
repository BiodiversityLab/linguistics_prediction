from utils.load_dataset import load_data, load_inference_data
from utils.load_json_configuration import LoadJSONConfiguration
from utils.module_loader import load_module
from utils.metadata_writer import write_dataset_metadata, write_run_parameters, get_feature_names, get_feature_names_test
from utils.plot import plot_loss, plot_mc_dropout, plot_feature_importance, plot_latitude_error
from utils.evaluation_helper import predict_mc_dropout, predict_latitude_error
from utils.default_functions import default_to_device, default_cat, default_permutation_importance
import argparse
import os
from torch import nn
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import mlflow
import geopandas as gpd
from shapely.geometry import Polygon
import torch
import numpy as np
import torchvision.transforms as transforms
import pdb
def main(args):

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exclude_channels = [int(channel) for channel in args.permutation_feature_importance_exclude_channels]
    configuration = LoadJSONConfiguration(os.path.join('architectures', args.architecture, 'configuration.json'), args.configuration)
    dataset_module = load_module('architectures.'+args.architecture+'.'+args.dataset_module)
    to_device_method = lambda X, y, device: default_to_device(X, X, device)

    if configuration.get('custom_to_device_method') is not None:
        to_device_method = dataset_module.to_device
    catenate_method = lambda x: default_cat(x)
    if configuration.get('custom_catenate_method') is not None:
        catenate_method = dataset_module.catenate
    selected_images = load_inference_data(configuration, args.feature_data_dir)

    feature_names = get_feature_names_test(selected_images, configuration)
    #load model
    model_module = load_module('architectures.'+args.architecture+'.' + args.model)
    input_channels = len(feature_names)
    if 'input_channels' in configuration:
        input_channels = configuration['input_channels']
    model = model_module.load_model(input_channels, configuration)
    model = model.to(device)
    model.load_state_dict(torch.load(args.weights ,map_location=device))

    transform = transforms.Compose([transforms.ToTensor()])
    prediction_polygons = gpd.GeoDataFrame({'geometry': [], 'predictions': []})
    
    for i in range(0, len(selected_images),128):
        partial_data = selected_images[i:i+128]
        dataset = dataset_module.SingleValueDataset(partial_data, np.zeros(len(partial_data)), transform=transform, configuration=configuration)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        model.eval()
        
        grid_cell_size = 25000

        for X, y, coordinates  in dataloader:
            X, y = to_device_method(X,y, device)
            #pdb.set_trace()
            pred = model(X)
            for index_coord, coord in enumerate(coordinates):
                x0 = int(coord.split('_')[1].split('.')[0])-(grid_cell_size/2)
                y0 = int(coord.split('_')[0].split('.')[0])-(grid_cell_size/2)
                x1 = x0 + grid_cell_size
                y1 = y0 + grid_cell_size
                polygon = Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
            # pdb.set_trace()
                prediction_polygons = prediction_polygons._append({'geometry': polygon, 'predictions': pred[index_coord].detach().cpu().numpy()[0]*20}, ignore_index=True)
        prediction_polygons.crs = "ESRI:54034"
        prediction_polygons.to_file('predictions.gpkg', driver="GPKG")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture', action='store', type=str, help='name of the model architecture to use')
    parser.add_argument('--model', action='store', type=str, help='name of the model class to use')
    parser.add_argument('--configuration',  action='store', type=str, help='the model configuration to be used')
    parser.add_argument('--dataset_module',  action='store', type=str, help='the dataset to be used')
    parser.add_argument('--epochs', action='store', type=int, help='number of epochs')
    parser.add_argument('--batch_size', action='store', type=int, help='batch size')
    parser.add_argument('--learning_rate', action='store', type=float, help='learning rate')
    parser.add_argument('--run_path', action='store', type=str, default='train', help='path to store the experiment')
    parser.add_argument('--mlflow_experiment', action='store', default='default')
    parser.add_argument('--permutation_feature_importance_exclude_channels', nargs='+', type=int, default=[], help='Set channels to not include in feature importance test')
    parser.add_argument('--weights', action='store', type=str, default='')
    parser.add_argument('--feature_data_dir', action='store', type=str, default='features')
    parser.add_argument('--multi_label', action='store_true')
    parser.add_argument('--device', action='store', type=str, default='0')

    args = parser.parse_args()

    main(args)