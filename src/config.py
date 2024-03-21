'''
Config file containing the parameters for plant detection, scaling and the perspective transformation.
'''

threshold_h_lower = 57/360
threshold_h_upper = 170/360
threshold_s_lower = 0.2
threshold_s_upper = 1.0
threshold_v_lower = 50
threshold_v_upper = 360 

errosion_iterations_default = 3
dilation_iterations_default = 13

errosion_iterations_search_range = (1,10)
dilation_iterations_search_range = (5,20)
# size notification at 
size_notify = 1000 # mm^2

# chessboard size
CB_SIZE = (8,5)
CB_MEASUREMENTS = (170,100)
FRAME_MEASUREMENTS = (546.0,347.0)
CORNER_OFFSET = (64,55)