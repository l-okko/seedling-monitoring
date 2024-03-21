import cv2
import numpy as np
import config
import scales

def get_perspective_transform(frame):
    """ Get the perspective transform for the chessboard in the frame

    _________________________________________________________________________
    Input:  frame (np.array) - The frame to get the perspective transform for
            chessboard_size (tuple) - The size of the chessboard
           
    Output: perspective_transform (np.array) - The perspective transform for the chessboard

    _________________________________________________________________________

    This function finds the corners of the chessboard in the frame and calculates the perspective transform
    for the chessboard. It then returns the perspective transform. It makes use of opencv's findChessboardCorners function.
    It uses the left lower corner of the chessboard as a fixed reference point and scales the real world coordinates from there.

    """
    # define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find chessboard corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, config.CB_SIZE, None)

    if ret:
        # might be a bit overkill to refine the corners, but it's good to have
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        h, w = frame.shape[:2]
        # find the outer corners of the chessboard
        bottom_right = np.array(corners_refined[0][0])
        bottom_left = np.array(corners_refined[config.CB_SIZE[0] - 1][0])
        top_right = np.array(corners_refined[-config.CB_SIZE[0]][0])
        top_left = np.array(corners_refined[-1][0])

        corners = np.array([bottom_right,bottom_left,top_right,top_left])

        top_left = corners[np.argmin([np.linalg.norm(cx - np.array([0,0])) for cx in corners])]
        top_right = corners[np.argmin([np.linalg.norm(cx - np.array([w,0])) for cx in corners])]
        bottom_left = corners[np.argmin([np.linalg.norm(cx - np.array([0,h])) for cx in corners])]
        bottom_right = corners[np.argmin([np.linalg.norm(cx - np.array([w,h])) for cx in corners])]
        # Define the 4 points of the chessboard in the image
        # As the chessboard is not filling the whole image and it is hard to guess what we need as real world points.
        # Therefore I went with the lower left corner as the origin and then went 5 squares to the left and 8 squares up 
        # using the min_distance
        
        image_points = np.array([top_left,  bottom_left, bottom_right, top_right ], dtype=np.float32)
     
        
        
        x_scale = w/config.FRAME_MEASUREMENTS[0]
        y_scale = h/config.FRAME_MEASUREMENTS[1]

        cb_width = x_scale*config.CB_MEASUREMENTS[0]
        cb_height = y_scale*config.CB_MEASUREMENTS[1]
        
         # Adjust scales in the scale file
        # TODO PUT THE CB_MEASUREMENTS IN THE CONFIG FILE
        # oder ist x_scale und y_scale pxl/mm?
        scales.x_scale_factor = x_scale
        scales.y_scale_factor = y_scale

        offset_width = x_scale*config.CORNER_OFFSET[0]
        offset_height = y_scale*config.CORNER_OFFSET[1]
        
        real_world_points = np.array([(offset_width,h - cb_height - offset_height),(offset_width,h- offset_height),(offset_width + cb_width,h - offset_height),(offset_width + cb_width,h- cb_height - offset_height)], dtype=np.float32)

        # Calculate the perspective transform
        perspective_transform = cv2.getPerspectiveTransform(image_points, real_world_points)

        return perspective_transform, [top_left,bottom_left,bottom_right,top_right]
    
    return None

def transform_image(frame, perspective_transform):
    """ Apply the perspective transform to the frame

    _________________________________________________________________________
    Input:  frame (np.array) - The frame to apply the perspective transform to
            perspective_transform (np.array) - The perspective transform to apply to the frame
           
    Output: warped_frame (np.array) - The frame with the perspective transform applied

    _________________________________________________________________________

    This function applies the perspective transform to the frame and returns the warped frame. It makes use of opencv's warpPerspective function.

    """
    # Apply the perspective transform
    warped_frame = cv2.warpPerspective(frame, perspective_transform, frame.shape[:2][::-1] )
    return warped_frame

def use_example():
    """ Use the get_perspective_transform function to capture images from the camera and apply the perspective transform to the images
    It Youses your webcam, so make sure to have one connected and it is in the correct number (Usually it is 0 or 1)
    
    Just google "OpenCV chessboard pdf" oder so und zeig das der Kamera. Dann sollte es funktionieren.
    """
    # Capture images from the camera
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Find chessboard corners
        chessboard_size = (8,5)
        perspective_transform = get_perspective_transform(frame, chessboard_size, (170,100), (64,55), (546,347))
        if perspective_transform is not None:
            print(perspective_transform)

            # Apply the perspective transform
            warped_frame = cv2.warpPerspective(frame, perspective_transform, frame.shape[:2][::-1] )
            cv2.imshow('Warped', warped_frame)

        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

#if __name__ == "__main__":
#    use_example()