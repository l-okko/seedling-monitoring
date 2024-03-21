import file_handler
from testing_linus import get_calibration_data
import cv2
import config
import matplotlib.pyplot as plt

def demo_perspective():
    config.CB_SIZE = (8,5)
    config.FRAME_MEASUREMENTS = (500,500)
    config.CORNER_OFFSET = (10,10)
    cal_image,_ = file_handler.load_image("demo/1.jpg")
    cv2.imwrite("demo_cal_1_cal.jpg",cal_image)
    pt = get_calibration_data(cal_image)
    image, time = file_handler.load_image('demo/1.jpg')
    h, w = image.shape[:2]
    if pt is not None:
        image = cv2.warpPerspective(image, pt, (w,h))
    
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(cal_image)
    ax[1].imshow(image)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[0].set_title("Original")
    ax[1].set_title("Warped")
    fig.savefig("demo/perspective_warp.png",dpi=300,pad_inches=0.1,bbox_inches='tight')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite("demo_cal_1.jpg",image)
    
if __name__== '__main__':
    demo_perspective()