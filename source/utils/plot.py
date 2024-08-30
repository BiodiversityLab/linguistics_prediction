import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import pdb
def plot_loss_stats(maeLoss,best_loss, save_path=None,name=''):
    #plt.figure(figsize=(4, 4))
    fig, ax = plt.subplots(figsize=(3,2))
    ax.axis('off')

    #plt.title("Loss Statistics")
    ax.text(0.0,0.9,"Loss Statistics", fontsize=14, va="center")
    x_pos = 0.0
    y_pos_mae = 0.3
    y_pos_mse = 0.1
    
    # Skala MAELoss och skriv ut den
    mae_text = f"Scaled MAELoss:"
    ax.text(0.0, y_pos_mae, mae_text, ha="left", va="center", transform=plt.gca().transAxes)
    ax.text(1, y_pos_mae, f"{maeLoss:.2f}", ha="right", va="center", transform=plt.gca().transAxes)

    # Skriv ut MSELoss
    mse_text = f"MSELoss: "
    ax.text(0.0, y_pos_mse, mse_text, ha="left", va="center", transform=plt.gca().transAxes)
    ax.text(1, y_pos_mse, f"{best_loss:.4f}", ha="right", va="center", transform=plt.gca().transAxes)
    plt.savefig(os.path.join(save_path,f'{name}.png'), format='png', dpi=150)


def plot_loss(training_losses, save_path=None,name=''):
    plt.figure(figsize=(10,5))
    plt.title("Training Loss per Epoch")
    plt.plot(training_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    if save_path:
        plt.savefig(os.path.join(save_path,f'loss{name}.png'), format='png', dpi=150)
        print(f"Plot saved as {save_path}")
    else:
        plt.show()

def plot_val_loss(val_losses, save_path=None,name=''):
    plt.figure(figsize=(10,5))
    plt.title("Val Loss per Epoch")
    plt.plot(val_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    if save_path:
        plt.savefig(os.path.join(save_path,f'loss{name}.png'), format='png', dpi=150)
        print(f"Plot saved as {save_path}")
    else:
        plt.show()

def plot_mc_dropout(labels, mean, std, save_path=None,  maximum_label=1000, name='', maximum_n_taxa=4):
        plt.figure(figsize=(6, 6))
        plt.errorbar(labels.flatten() , mean.flatten() , yerr=std.flatten() , fmt='.', alpha=1, ecolor='black', elinewidth=0.5)
        plt.xlim(0,  maximum_n_taxa)
        plt.ylim(0,  maximum_n_taxa)
        plt.plot([0,  maximum_n_taxa], [0,  maximum_n_taxa], 'r-')
        plt.xlabel('True diversity')
        plt.ylabel('Predicted diversity')
        plt.grid()
        plt.tight_layout()
        now = datetime.now()
        date_time = now.strftime("%y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_path,'plot_mc_dropout_'+date_time+name+'.png'), bbox_inches='tight', dpi=150)
        plt.close()

def plot_feature_importance(importance_scores, configuration, experiment_path, feature_names,name=''):
    plt.figure(figsize=(12, 10))
 
    plt.bar(feature_names, importance_scores)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
            #plt.figure(figsize=(20, 20))
    plt.savefig(os.path.join(experiment_path,f'feature_importance{name}.png'))

def plot_latitude_error(mean, latitude, save_path=None):
    plt.figure(figsize=(6, 6))
    coefficients = np.polyfit(latitude.flatten(),  mean.flatten(), 1)

    linear_poly = np.poly1d(coefficients)
    x_values = [np.min(latitude.flatten()), np.max(latitude.flatten())]
    y_values = linear_poly(x_values)
    plt.plot(x_values, y_values)


    plt.errorbar(latitude.flatten(), mean.flatten(), yerr=latitude.flatten(), fmt='.', alpha=1, ecolor='black', elinewidth=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, mean.max())

    plt.xlabel('latitude')
    plt.ylabel('error')
    plt.grid()
    plt.tight_layout()

    plt.xlabel('latitude')
    plt.ylabel('error')
    plt.grid()
    plt.tight_layout()
    now = datetime.now()
    date_time = now.strftime("%y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_path,'plot_latitude_error_'+date_time+'.png'), bbox_inches='tight', dpi=150)
    plt.close()



