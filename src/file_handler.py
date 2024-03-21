'''
Module for importing images and exporting csv files
'''

import image_correction
import cv2 as cv
from exif import Image
import logging
from datetime import datetime, timedelta
from plant import Plant
import numpy as np

def load_image(path):
    '''
    Loads the image with the path path
    Parameters:
        path (String): path of the image
    Returns:
        image : numpy array containing the image data
        timestamp (datetime): timestamp of when the picture was taken
    '''
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    w, h = image.shape[:2]
    image = cv.resize(image, (h//2, w//2), interpolation= cv.INTER_LINEAR)
    logging.debug(f"Loaded image: {path}")
    return image, timestamp(path)

def timestamp(path):
    '''
    Reads the datetime from the image exif data.
    
    Parameters:
        path (String): path of the image
    Returns:
        datetime (datetime): time when the picture was taken
    '''
    date_format = '%Y:%m:%d %H:%M:%S'
    
    with open(path, 'rb') as image_file:
        image = Image(image_file)
    if not image.has_exif or not 'datetime' in dir(image):
        logging.error("Image does not have a timestamp")
        return 0
    dt = datetime.strptime(image.datetime, date_format)
    logging.debug(f"Image {path} has timestamp {dt}")
    return dt

def save_history(plant: 'Plant',path):
    '''
    Save a .csv file containing all datapoints for one plant.
    
    Paramters:
        plant: plant object
        path: path where the file is saved
    '''
    logging.debug("Generating .csv file")
    hist = plant.get_history()
    csv = []
    for h in hist:
        csv.append([h[0],h[1]['origin'][0],h[1]['origin'][1],h[1]['area']])
    np.savetxt(path,csv,delimiter=',',fmt='%s',comments='',header="DATE,X,Y,AREA")
    logging.info(f"Saved .csv file: {path}")
    
def save_plants(plants,path):
    '''
    Save .csv file conatining information about all plants.
    
    Parameters:
        plants: Array containing all plant objects
        path: path where the file is saved
    '''
    csv = []
    for plant in plants:
        csv.append([plant.get_id(),plant.get_datetime_of_creation(),plant.get_history().__len__(),plant.get_germination_time().days])
    np.savetxt(path,csv,delimiter=',',fmt='%s', comments='',header="ID,FIRST DETECTED,NUM OF DATAPOINTS,GERMINATION TIME")
    logging.info(f"Saved .csv file: {path}")