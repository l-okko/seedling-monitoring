'''
Module and class to manage plants and plant objects
'''


import numpy as np
import logging
from datetime import datetime, timedelta

from scales import convert_pixels_to_area

class Plant:
    '''
    A class to represent a plant.
    '''
    
    _next_id = 0
    
    def new_id():
        id = Plant._next_id
        Plant._next_id += 1
        return id
    
    def __init__(self,time=datetime.now()) -> None:
        self.id = Plant.new_id()
        self.name = None
        self.green_area = 0
        self.health = 100
        self.origin = None
        self.area_pxls = 0
        self.area = 0 # in mm^2
        self.created = time
        self.germination = 0
        self.history = []
        self.det = False
        
    def get_id(self) -> int:
        return self.id
        
    def get_origin(self):
        return self.origin
        
    def set_origin(self,x,y) -> None:
        '''Set the coordinate origin of a plant'''
        self.origin = np.array([x,y])

    def get_health(self) -> int:
        '''Get health of this plant'''
        return self.health
    
    def set_health(self, health) -> None:
        '''Set health of this plant'''
        self.health = health

    def set_area_pxls(self, area_pxls, scale) -> None:
        self.area_pxls = area_pxls
        # when setting the pixel area, also set the area in mm^2
        self.area = np.round(convert_pixels_to_area(area_pxls,scale),1)
    
    def get_area_pxls(self) -> int:
        return self.area_pxls
    
    def set_area(self, area) -> None:
        self.area = area
    
    def get_area(self) -> int:
        return self.area
    
    def get_age(self) -> timedelta:
        return datetime.now() - self.created
    
    def get_datetime_of_creation(self) -> datetime:
        return self.created
    
    def log(self,timestamp) -> None:
        entry = [timestamp,{'origin':self.origin,'area':self.area}]
        self.history.append(entry)
        
    def update(self,new: 'Plant') -> None:
        self.area = new.get_area()
        self.area_pxls = new.get_area_pxls()
        self.origin = new.get_origin()
        self.log(new.get_datetime_of_creation())
        
    def compare(self,plant: 'Plant'):
        return np.linalg.norm(self.get_origin() - plant.get_origin())
    
    def get_history(self):
        return self.history
    
    def calc_germination_time(self, seed_time) -> None:
        self.germination = self.created - seed_time
    def get_germination_time(self) -> timedelta:
        return self.germination
    
    def detected(self) -> bool:
        return self.det
    
    def set_detected(self, b: bool) -> None:
        self.det = b

def map_plants(old: 'Plant', new: 'Plant', radius: int):
    create = []
    for new_plant in new:
        exists = False
        min_dist = 999999
        min_idx = None
        for idx, old_plant in enumerate(old):
            dist = old_plant.compare(new_plant)
            if dist < min_dist and dist < radius and not old_plant.detected():
                min_dist = dist
                min_idx = idx
                exists = True
        if not exists:
            logging.debug("Created new plant")
            create.append(new_plant)
        else:
            old[min_idx].update(new_plant)
            old[min_idx].set_detected(True)
    old = old + create
    old = undetected(old)
    return old
    
def find_plant(plants, id: int) -> Plant:
    '''
    Find plant with the given id.
    '''
    for plant in plants:
        if plant.get_id() == id:
            return plant
    return None

def calculate_germination_times(plants, time) -> None:
    for plant in plants:
        plant.calc_germination_time(time)
        
def undetected(plants):
    for plant in plants:
        plant.set_detected(False)
    return plants