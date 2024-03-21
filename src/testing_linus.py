
import file_handler
import cv2
import image_correction
from plant import Plant, map_plants, find_plant, calculate_germination_times
import matplotlib.pyplot as plt
import logging
import plant_detection
import os
import visualizer
from datetime import datetime

from config import errosion_iterations_default, dilation_iterations_default, threshold_h_lower, threshold_h_upper, threshold_s_lower, threshold_s_upper, threshold_v_lower, threshold_v_upper
import scales

logging.basicConfig(level=logging.INFO)

errosion_iterations = errosion_iterations_default
dilation_iterations = dilation_iterations_default

def get_calibration_data(image):
    perspective_transform, corners = image_correction.get_perspective_transform(image)
    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title("Automatically detected corners")
    for i in corners:
        ax.scatter(i[0],i[1],marker='x')
    fig.savefig('plots/corners.png',dpi=300)
    return perspective_transform

def analyze_image(image, datetime, perspective_transform = None):
    w, h = image.shape[:2]
    if perspective_transform is not None:
        image = cv2.warpPerspective(image, perspective_transform, (h,w))
    else:
        logging.info("No perspective transformation is applied")
    plants = plant_detection.get_plants_from_rgb(image,time, errosion_iterations, dilation_iterations)
    return image, plants

def show(image):
    cv2.imshow("Test",image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
if __name__=='__main__':
    # List all images in images folder
    fp = '/mnt/e/Syncthing/MIP/timelapse/all/'
    #fp = ''
    paths = os.listdir(fp)[0::2]
    paths = os.listdir(fp)[0:1]
    #paths = ['20240317_102012jpg.jpg']
    for idx,path in enumerate(paths):
        paths[idx] = fp + path
    # Sort images with the timestamp from exif data
    paths = sorted(paths,key=file_handler.timestamp)
    plants = []
    
    # Calc perspective transform
    cal_image,_ = file_handler.load_image('cal.jpg')
    pt = get_calibration_data(cal_image)
    
    # Analyze all images in paths
    for idx,path in enumerate(paths):
        logging.info(f"Analyzing {path}")
        image, time = file_handler.load_image(path)
        image, new_plants = analyze_image(image, time, pt)
        # Check if the detected plants can be mapped to existing ones
        # Relative movement of the origin by 50 px is allowed
        plants = map_plants(plants, new_plants, 50)
        plants = sorted(plants,key=Plant.get_id)
        #visualizer.save_plant_img(image, plants[0:5], f"hist/result_pd{idx}.png")
        
        #plant = find_plant(plants,91)
        #if plant is not None:
        #    visualizer.save_single_plant(image, plant,250,f"tl/{idx}.png")
    
    calculate_germination_times(plants,datetime(year=2024,month=3,day=8,hour=12,minute=0,second=0))
    for idx,plant in enumerate(plants):
        file_handler.save_history(plant,f"log/plant_{plant.get_id()}.csv")
    file_handler.save_plants(plants,"plants.csv")
    #visualizer.save_plant_history(find_plant(plants,91),"hist.png",'area')
    visualizer.save_plant_img(image, plants, "result_pd.png")
    cv2.imwrite("result.jpg",image)


 
