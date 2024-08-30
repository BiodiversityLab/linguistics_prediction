from sklearn.ensemble import RandomForestRegressor
from torch import nn


class CustomRandomForestRegressor(nn.Module):
   def __init__(self, input_channels):
      self.random_forest_regressor = RandomForestRegressor()

   def to(self, device):
      return self.random_forest_regressor

def load_model(input_channels, configuration):
   randomForestRegressor = CustomRandomForestRegressor(input_channels)
   return randomForestRegressor
