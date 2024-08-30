import torch
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

def default_to_device(X, y, device):
    return X.to(device), y.to(device)
  

def default_cat(x):
    return torch.cat(x)


def default_permutation_importance(model, X, y, configuration, feature_names, exclude_channels=[]):

    model.eval()

    #feature_names = configuration['features']
    baseline_performance = mean_squared_error(y.cpu(), model(X).cpu().detach().numpy())
    importance_scores = []

    for i in range(X.shape[1]): #for loop in feature dimension
        if i in exclude_channels:
            importance_scores.append(0)
            continue

        X_permuted = X.clone().detach()
        X_permuted[:, i] = shuffle(X_permuted[:, i])
        permuted_performance = mean_squared_error(y.cpu(), model(X_permuted).cpu().detach().numpy())
        importance = abs(baseline_performance - permuted_performance)
        importance_scores.append(importance)

    adjusted_importance_scores = [score for i, score in enumerate(importance_scores) if i not in exclude_channels]
    adjusted_feature_names = [name for i, name in enumerate(feature_names) if i not in exclude_channels]
    return adjusted_importance_scores, adjusted_feature_names


