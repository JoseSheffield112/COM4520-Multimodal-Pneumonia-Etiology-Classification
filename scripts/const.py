from enum import Enum


class Models(Enum):
     static = 1
     time_series = 2
     image_data = 3
     static_and_time_series = 3
     static_and_time_series_and_image_data = 4
     static_and_image_data = 5
