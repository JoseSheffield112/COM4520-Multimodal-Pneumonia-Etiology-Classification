from enum import Enum

nr_static_features = 12
nr_timeseries_features = 3
image_encoder_output_size = 1024

class Models(Enum):
     static = 1
     timeseries = 2
     image = 3
     static_timeseries = 4
     static_timeseries_image = 5
     static_image = 6
     timeseries_image = 7