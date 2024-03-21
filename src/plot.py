'''
Load csv files and create plots
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime
import os
import file_handler
import pickle 
import cv2
import testing_linus
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

FOLDER_CSV = 'log/'
FOLDER_IMAGES = '/mnt/e/Syncthing/MIP/timelapse/all/'
LOOKUP = {}

cal_image,_ = file_handler.load_image('cal.jpg')
pt = testing_linus.get_calibration_data(cal_image)

def generate_timestamp_lookup():
    paths = os.listdir(FOLDER_IMAGES)
    for idx,path in enumerate(paths):
        time = file_handler.timestamp(FOLDER_IMAGES + path)
        LOOKUP.update({time:path})
        print(path)

def plot_area_history(id: int):
    csv = pd.read_csv(f'{FOLDER_CSV}plant_{id}.csv')
    df = pd.DataFrame(csv)
    df['DATE'] = pd.to_datetime(df['DATE'], format="%Y-%m-%d %H:%M:%S")
    print(df['DATE'])
    fig, ax = plt.subplots(1,1)
    ax.plot(df['DATE'],df['AREA'],marker='x',ls='')
    #print(np.argmin(df['AREA'][30:60]))
    myFmt = mdates.DateFormatter('%d.%b')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_ylabel(r'Area [${mm}^2$]')
    ax.set_xlabel("time")
    ax.set_title(f"Total green area of plant with id: {id}")
    filename = f"plots/area_plot_{id}.png"
    fig.savefig(filename,dpi=300,bbox_inches="tight",pad_inches=0.1)
    
def annotate(id: int, idx):
    csv = pd.read_csv(f'{FOLDER_CSV}plant_{id}.csv')
    df = pd.DataFrame(csv)
    df['DATE'] = pd.to_datetime(df['DATE'], format="%Y-%m-%d %H:%M:%S")
    fig, ax = plt.subplots(1,1)
    ax.plot(df['DATE'],df['AREA'],marker='x',ls='')
    image = cv2.imread(FOLDER_IMAGES + LOOKUP[df['DATE'][idx]])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    w, h = image.shape[:2]
    image = cv2.resize(image, (h//2, w//2), interpolation= cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    imagebox = OffsetImage(image, zoom=0.2)
    imagebox.image.axes = ax

    ab = AnnotationBbox(imagebox, (df['DATE'][idx],df['AREA'][idx]),
                        xybox=(120., -80.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        )

    ax.add_artist(ab)
    myFmt = mdates.DateFormatter('%d.%b')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_ylabel(r'Area [${mm}^2$]')
    ax.set_xlabel("time")
    ax.set_title(f"Total green area of plant with id: {id}")
    filename = f"plots/area_plot_v2_{id}.pdf"
    fig.savefig(filename,dpi=300,bbox_inches="tight",pad_inches=0.1)
    
def show_plant(id,timestamp,size,idx):

    csv = pd.read_csv(f'{FOLDER_CSV}plant_{id}.csv')
    df = pd.DataFrame(csv)
    df['DATE'] = pd.to_datetime(df['DATE'], format="%Y-%m-%d %H:%M:%S")
    row = df.loc[df['DATE'] == timestamp]
    
    image = cv2.imread(FOLDER_IMAGES + LOOKUP[timestamp])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    w, h = image.shape[:2]
    image = cv2.resize(image, (h//2, w//2), interpolation= cv2.INTER_LINEAR)
    w, h = image.shape[:2]
    if pt is not None:
        image = cv2.warpPerspective(image, pt, (h,w))
    w, h = image.shape[:2]
    xy = [row['X'],row['Y']]
    print(row['X'])
    x = int(xy[1])
    y = int(xy[0])
    if x - size <= 0 or x + size >= w or y - size <= 0 or y + size >= h:
        image = cv2.copyMakeBorder(image,size,size,size,size,cv2.BORDER_CONSTANT,0)
        x -= size
        y -= size
        h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'plots/plant_image_{id}_{idx}.jpg',image[x-size:x+size,y-size:y+size,:])
    
def overview():
    plants_csv = pd.read_csv('plants.csv')
    plants_df = pd.DataFrame(plants_csv)
    pos_x = []
    pos_y = []
    ids = []
    for plant in plants_df['ID']:
        csv = pd.read_csv(f'{FOLDER_CSV}plant_{id}.csv')
        df = pd.DataFrame(csv)
        lx = df['X'][-1]
        ly = df['Y'][-1]
        ids.append(id)
        pos_x.append(lx)
        pos_y.append(ly)

    
if __name__=='__main__':
    id = 60
    plot_area_history(id)
    with open('lookup.pkl', 'rb') as f:
        LOOKUP = pickle.load(f)
    csv = pd.read_csv(f'{FOLDER_CSV}plant_{id}.csv')
    df = pd.DataFrame(csv)
    df['DATE'] = pd.to_datetime(df['DATE'], format="%Y-%m-%d %H:%M:%S")
    print(df['Y'][10])
    for i in range(34):
        show_plant(id,df['DATE'][i],100,i)
    #annotate(id,6)
