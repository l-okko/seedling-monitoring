'''
Creating overlays and plots with plant data
'''

import matplotlib.pyplot as plt
import numpy as np
from scales import check_area
from plant import Plant

import cv2

def save_plant_img(img, plants, name, color_marker = 'r' ,color_text = 'b', color_circle = 'black', field = None):
    """ Mark found Plants and save image as name"""
    fig, ax = plt.subplots(1, 1 )
    ax.imshow(img, alpha= 0.7)

    for plant in plants:
        if field is None:
            center = plant.get_origin()
            area = plant.get_area()
            first_detected = plant.get_datetime_of_creation()
            id = plant.get_id()
            
            try:
                time_str = first_detected.strftime("%d-%m-%Y, %H:%M:%S")
            except AttributeError:
                time_str = "unknown"
                Warning("No time found - this might be intentional")
            string = f"id: {id}\narea: {area}\ndate: {time_str}"
            #string = f"id:{id}"
        else:
            string = str(plant.field)
        
        ax.scatter(center[0], center[1], c=color_marker, marker='x')
        ax.text(center[0], center[1], string, c=color_text, size=7.0)
        
        # check if the area is above the size notification
        if check_area(plant.get_area()):
            # draw a circle with the area as radius
            circle = plt.Circle(center, np.sqrt(area/np.pi), color=color_circle, fill=False)
            ax.add_artist(circle)
    
    #hide text
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    fig.savefig(name, dpi = 300, bbox_inches='tight', pad_inches=0.0)
    
    
def save_single_plant(img, plant, size: int,path):
    '''
    Save a section of the image only containg the specified plant
    
    Parameters:
        img: image from wich the sectionis cut
        plant: plant that should be in the image
        size: size of the cutout
        path: filepath where the image is saved
    '''
    h, w = img.shape[:2]
    xy = plant.get_origin()
    x = int(xy[0])
    y = int(xy[1])
    if x - size <= 0 or x + size >= w or y - size <= 0 or y + size >= h:
        img = cv2.copyMakeBorder(img,size+2,size+2,size+2,size+2,cv2.BORDER_CONSTANT,0)
        x += size
        y += size
        h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,img[x-size:x+size,y-size:y+size,:])
    
def save_plant_history(plant, name, parameter):
    history = plant.get_history()
    fig, ax = plt.subplots(1,1)
    series = []
    for h in history:
        series.append([h[0],np.linalg.norm(h[1][parameter])])
    ax.plot(*zip(*series))
    fig.tight_layout()
    fig.savefig(name, dpi = 300, bbox_inches='tight', pad_inches=0.0)
