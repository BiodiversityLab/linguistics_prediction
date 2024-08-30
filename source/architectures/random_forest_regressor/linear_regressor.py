from sklearn.linear_model import LinearRegression
from torch import nn


class CustomLinearRegressor(nn.Module):
   def __init__(self, input_channels):
      self.random_forest_regressor = LinearRegression()

   def to(self, device):
      return self.random_forest_regressor

def load_model(input_channels, configuration):
   randomForestRegressor = CustomLinearRegressor(input_channels)
   return randomForestRegressor
