import torch
import numpy as np
import pdb
def predict_mc_dropout(dataloader, model, device, num_samples, to_device_method,  maximum_label):

    model.train()  # Set model to train mode
    model.to(device)
    predictions = []
    labels = []
    for _ in range(num_samples):
        with torch.no_grad():
            sample_predictions = []
            for X, y, coordinates in dataloader:
                X, y = to_device_method(X,y, device)
                # Compute prediction error for this sample
      
                pred = model(X)

                sample_predictions.extend(pred.detach().cpu().numpy())

            predictions.append(sample_predictions)

    for X, y, coordinates in dataloader:
        labels.extend(y.detach().cpu().numpy())
  #  pdb.set_trace()
    # Calculate mean and standard deviation of predictions
    avg_predictions = np.mean(np.stack(predictions), axis=0)
    std_predictions = np.std(np.stack(predictions), axis=0)
    avg_predictions *=  maximum_label
    std_predictions *=  maximum_label
    labels = np.array(labels)
    labels *=  maximum_label
    return avg_predictions, std_predictions, labels


def get_lat_values(X, index_of_lat):
    lats = []
    for sample in X:
     
        lats.append(sample[index_of_lat].cpu())
    return lats

    
def predict_latitude_error(feature_names, dataloader, model, device, num_samples, to_device_method,  maximum_label):
    '''
    Currently only working with dataset that has format:
    X['constants'] == tensor array with single values
    X == tensor array with single values

    '''
    model.train()  # Set model to train mode
    predictions = []
    errors = []
    labels = []
    lats = []
    index_of_lat = feature_names.index('lat')
    for _ in range(num_samples):
        with torch.no_grad():
            sample_predictions = []
            sample_errors = []
            for X, y, coordinates in dataloader:
                X, y = to_device_method(X,y, device)
                pred = model(X)
                error = abs(pred-y)
                sample_errors.extend(error.cpu())
                sample_predictions.extend(pred.cpu())
            errors.append(sample_errors)
            predictions.append(sample_predictions)
    for X, y in dataloader:
        if type(X)==dict:
            lats.extend(get_lat_values(X['constants'], index_of_lat))
        else:
            lats.extend(get_lat_values(X, index_of_lat))
        labels.extend(y.cpu())
    avg_errors = np.mean(np.stack(errors), axis=0)

    avg_errors *=  maximum_label

    return avg_errors, np.array(lats)
