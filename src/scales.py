# a class to adjust things to real world values
from config import size_notify
from config import CB_MEASUREMENTS
from config import FRAME_MEASUREMENTS

import numpy as np

def scale(h,w):
    return np.array([w/FRAME_MEASUREMENTS[0],h/FRAME_MEASUREMENTS[1]])

def convert_pixels_to_area(num_pixels,scale: np.ndarray):
    """ Converts the number of pixels to the area in mm^2 """
    pxl_volume_scale = np.linalg.norm(scale)
    return num_pixels / pxl_volume_scale


def check_area(area):
    """ Checks if the area is above the size notification """
    if area > size_notify:
        return True
    return

def check_plants_area(plants):
    """ Checks if the area of the plants is above the size notification """
    # array the size of the plants
    notify = []

    #check if plants are empty
    if not plants:
        Warning("No plants found")
    
    for plant in plants:
        notify.append(check_area(plant.get_area()))
        
    return notify