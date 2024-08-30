from utils.load_dataset import load_data
from utils.load_json_configuration import LoadJSONConfiguration
from utils.module_loader import load_module
from utils.metadata_writer import write_dataset_metadata, write_run_parameters, get_feature_names
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
    

def warmup_lr_lambda(current_step, warmup_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

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
    features, labels, target_df, maximum_label = load_data(configuration, args.feature_data_dir, args.multi_label)
    write_dataset_metadata(features, dir_name, configuration)
    feature_names = get_feature_names(features, configuration)

    
    write_run_parameters(args, dir_name)
    
    
    train_dataset, val_dataset, test_dataset = dataset_module.prepare_dataset(features, labels, target_df, configuration)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    #load the model
    model_module = load_module('architectures.'+args.architecture+'.' + args.model)
    input_channels = len(feature_names)
    if 'input_channels' in configuration:
        input_channels = configuration['input_channels']
    model = model_module.load_model(input_channels, configuration)
    model = model.to(device)
    if args.weights != '':
        model.load_state_dict(torch.load(args.weights ,map_location=device))

    warmup_steps = 1000
    lr = args.learning_rate
    l1loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps))
    early_stopper = EarlyStopper(patience=20, min_delta=0.01)
    
    epochs = args.epochs
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    # Train the model
    start = time.time()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        
       
        train_loss = train(train_dataloader, model, loss_fn, optimizer, scheduler, to_device_method, maximum_label, device, t)
        train_losses.append(train_loss)
        mlflow.log_metric("train_loss", train_loss, step=t)

        val_MSEloss, val_MAEloss  = test(val_dataloader, model, l1loss_fn, loss_fn, device, configuration, feature_names, exclude_channels, dir_name, to_device_method, catenate_method, permutation_importance_method, maximum_label)

        mlflow.log_metric("val_MSEloss", val_MSEloss, step=t)
        mlflow.log_metric("val_MAEloss", val_MAEloss, step=t)
        
        if early_stopper.early_stop(val_MSEloss):
            break
        # scheduler.step()
        if val_MSEloss < best_loss:
            best_loss = val_MSEloss
            torch.save(model.state_dict(), os.path.join(dir_name,'model_weights.pth'))
            mlflow.log_artifact(os.path.join(dir_name,'model_weights.pth'), "weights")
    stop = time.time()
    
    print(f"Done in {stop - start:0.2f} seconds!")
    if len(test_dataloader) == 0:
        test_dataloader = val_dataloader
    test_MSEloss, test_MAEloss = test(test_dataloader, model, l1loss_fn, loss_fn, device, configuration, feature_names, exclude_channels, dir_name, to_device_method, catenate_method, permutation_importance_method, maximum_label, final_epoch=True)
    mlflow.log_metric("test_MSEloss", test_MSEloss, step=t)
    mlflow.log_metric("test_MAEloss", test_MAEloss, step=t)
    mlflow.log_metric("validation length", len(val_dataloader)*args.batch_size)
    mlflow.log_metric("test length", len(test_dataloader)*args.batch_size)

    plot_loss(train_losses, save_path=dir_name)

    for filename in os.listdir(dir_name):
        if filename.endswith('.png'):
            file_path = os.path.join(dir_name, filename)
            mlflow.log_artifact(file_path)

    create_gpkg_predictions(model, train_dataloader, "train", dir_name, to_device_method, device, maximum_label)
    create_gpkg_predictions(model, val_dataloader, "validation", dir_name, to_device_method, device, maximum_label)
    create_gpkg_predictions(model, test_dataloader, "test", dir_name, to_device_method, device, maximum_label)


def create_gpkg_predictions(model, dataloader, name, dir_name, to_device_method, device, maximum_label):
    prediction_polygons = gpd.GeoDataFrame({'geometry': [], 'predictions': []})
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
            prediction_polygons = prediction_polygons.append({'geometry': polygon, 'predictions': pred[index_coord].detach().cpu().numpy()[0]*maximum_label}, ignore_index=True)
    prediction_polygons.crs = "ESRI:54034"
    prediction_polygons.to_file(os.path.join(dir_name,f'{name}_predictions.gpkg'), driver="GPKG")


def create_run_dir(run_basepath: Path):
    index = 0
    dir_name = run_basepath

    while os.path.exists(dir_name):
        index += 1
        dir_name = f"{run_basepath}_{index}"

    os.makedirs(dir_name)
    return dir_name


def train(dataloader, model, loss_fn, optimizer, scheduler, to_device_method, maximum_label, device, epoch):
    size = len(dataloader.dataset)
    print("size of dataset")
    print(size)
    model.train()
    train_loss = 0
    for batch, (X, y, coordinates ) in enumerate(dataloader):
        X, y = to_device_method(X,y, device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        if epoch == 0:
            continue
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
       
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= len(dataloader)
    print(f"Average train loss: {train_loss:>8f} \n")
    return train_loss


def test(dataloader, model, l1loss_fn, loss_fn, device, configuration, feature_names, exclude_channels, experiment_path, to_device_method, catenate_method, permutation_importance_method, maximum_label, final_epoch=False):
    model.eval()
    MSEloss = 0
    MAEloss = 0
    val_loss = 0
    l1_val_loss = 0
    X_samples = []
    y_samples = []

    prediction_polygons = gpd.GeoDataFrame({'geometry': [], 'labels': [], 'predictions': []})
    grid_cell_size = 25000

    with torch.no_grad():
        for X, y, coordinates  in dataloader:
            X, y = to_device_method(X,y, device)
            
            X_samples.append(X)
            y_samples.append(y)
            # Compute prediction error
            pred = model(X)
            for index_coord, coord in enumerate(coordinates):
                x0 = int(coord.split('_')[1].split('.')[0])-(grid_cell_size/2)
                y0 = int(coord.split('_')[0].split('.')[0])-(grid_cell_size/2)
                x1 = x0 + grid_cell_size
                y1 = y0 + grid_cell_size
                polygon = Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
                
            pred_val = pred[index_coord].detach().cpu().numpy()

            if len(pred_val.shape)>2:
                prediction_polygons = prediction_polygons.append({'geometry': polygon, 'labels': y[index_coord].detach().cpu().numpy()[0]*maximum_label, 'predictions': pred_val[0][0][0]*maximum_label}, ignore_index=True)
            elif len(pred_val.shape)>1:
                prediction_polygons = prediction_polygons.append({'geometry': polygon, 'labels': y[index_coord].detach().cpu().numpy()[0]*maximum_label, 'predictions': pred_val[0][0]*maximum_label}, ignore_index=True)
            else:
                prediction_polygons = prediction_polygons.append({'geometry': polygon, 'labels': y[index_coord].detach().cpu().numpy()[0]*maximum_label, 'predictions': pred_val[0]*maximum_label}, ignore_index=True)


            MSEloss += loss_fn(pred, y).item()
            MAEloss += l1loss_fn(pred, y).item()


    MSEloss /= len(dataloader)
    MAEloss /= len(dataloader)
    MAEloss *= maximum_label
    print(f"Average valid loss: {MSEloss:>8f} \n")
    print(f"Average rescaled l1 valid loss: {MAEloss:>8f} \n")

    X_sampled = catenate_method(X_samples)
    
    y_sampled = torch.cat(y_samples)
    #Permutation feature importance
    if final_epoch:
        torch.save(model.state_dict(), os.path.join(experiment_path,'last_model_weights.pth'))
        mlflow.log_artifact(os.path.join(experiment_path,'last_model_weights.pth'), "weights")
        #pdb.set_trace()
        prediction_polygons.crs = "ESRI:54034"
        prediction_polygons.to_file('predictions.gpkg', driver="GPKG")
       # device = torch.device('cpu')
        mc_dropout_mean, mc_dropout_std, mc_labels = predict_mc_dropout(dataloader, model, device, num_samples=100, to_device_method=to_device_method,  maximum_label=maximum_label)
        if args.multi_label:
           
            plot_mc_dropout(mc_labels[:,:,0], mc_dropout_mean[:,0], mc_dropout_std[:,0], save_path=experiment_path,  maximum_label=maximum_label,name='_overlapped')

            plot_mc_dropout(mc_labels[:,:,1], mc_dropout_mean[:,1], mc_dropout_std[:,1], save_path=experiment_path,  maximum_label=maximum_label,name='_large_overlapped')

        else:
            plot_mc_dropout(mc_labels, mc_dropout_mean, mc_dropout_std, save_path=experiment_path,  maximum_label=maximum_label)

       # avg_errors, lats = predict_latitude_error(feature_names, dataloader, model, device, num_samples=100, to_device_method=to_device_method,  maximum_label= maximum_label)   
        #plot_latitude_error( avg_errors, lats, save_path=experiment_path)
    
        importance_scores, feature_names = permutation_importance_method(model, X_sampled, y_sampled, configuration, feature_names, exclude_channels=exclude_channels)
        with open("feature_importance.txt", "w") as f:
            for score in importance_scores:
                f.write(f"{score}\n")
        mlflow.log_artifact("feature_importance.txt")
        
        plot_feature_importance(importance_scores, configuration, experiment_path, feature_names)
        
    return MSEloss, MAEloss
    #return val_loss, l1_val_loss


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
