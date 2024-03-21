import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as ndi
import plant
from datetime import datetime
import time
import cv2

from config import threshold_h_lower, threshold_h_upper, threshold_s_lower, threshold_s_upper, threshold_v_lower, threshold_v_upper, errosion_iterations_default, dilation_iterations_default
from config import dilation_iterations_search_range, errosion_iterations_search_range
from scales import check_area, scale

def get_hsv(image):
    """Split the image (RGB) into H, S, V channel
    ________________________________________________
    image 
    """

    hsv = colors.rgb_to_hsv(image)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    return h, s, v


def threshold_image(h, s, v):
    """
        Create a mask using thresholds of the hsv channels

        ____________________________________________________
        h 
        s
        v 

        thresholds"""
    # Create an empty mask
    mask = np.empty_like(h, dtype=bool)
    # start with a true output
    out = np.ones_like(h, dtype=bool)

    # Perform thresholding in-place
    np.logical_and(np.less(threshold_h_lower, h, out=mask), out, out=out)
    np.logical_and(np.less(h, threshold_h_upper, out=mask), out, out=out)
    np.logical_and(np.less(threshold_s_lower, s, out=mask), out, out=out)
    np.logical_and(np.less(s, threshold_s_upper, out=mask), out, out=out)
    np.logical_and(np.less(threshold_v_lower, v, out=mask), out, out=out)
    np.logical_and(np.less(v, threshold_v_upper, out=mask), out, out=out)
    return out




def cluster_mask(mask, errosion_iterations, dilation_iterations):
    """ Erode and dilate the mask, if the mask is not boolean it will be transformed
        First erodes the mask to remove noise, then dilates it.
        Uses scipy.ndimage.label()
    ____________________________________________________
    mask Maske
    erosion iteration
    
    returns:

    clustered mask, number of clusters
    """
    # check if the mask is of type bool
    if mask.dtype != bool:
        mask = mask.astype(bool)
    
    mask_eroded = ndi.binary_erosion(mask, iterations=errosion_iterations)
    mask_dilated = ndi.binary_dilation(mask_eroded, iterations=dilation_iterations)
    mask_clusters, mask_n_clusters = ndi.label(mask_dilated)
    return mask_clusters, mask_n_clusters


def get_center_and_areas(mask, clustered_mask, n_clusters):
    """Assuming that clustered mask is not smaller at plant positions
    counts the identfied pixels (green pixels) from the mask, inside the clustered mask.
    Todo: somethong more suffisticated?

    returns:
    centers
    areas
    """

    areas = []
    centers = []

    if mask.dtype != bool:
        mask = mask.astype(bool)
    # erode and dialte mask once to remove pixel level (small) noise
    # without realy effecting coherent areas
    mask = ndi.binary_erosion(mask, iterations=1)
    mask = ndi.binary_dilation(mask, iterations=1)
    
        # Count the pixels inside the eroded mask for each cluster
    for i in range(1, n_clusters+1):
        # Find indices where clustered_mask equals i and mask is True
        y, x = np.where((clustered_mask == i) & mask)
        
        # Count the number of true pixels in the cluster
        area = len(y)
        
        # Calculate the centroid of the true pixels in the cluster
        if area > 0:
            center = (np.mean(x), np.mean(y))
            centers.append(center)
        else:
            centers.append(None)
        
        # Append area to the areas list
        areas.append(area)
    return centers, areas

def get_centers_and_areas_from_rgb(img, errosion_iterations, dilation_iterations):
    """What would you expect?
        Area in pxls """

    # get the hsv channels
    h, s, v = get_hsv(img)
    # determine mask
    mask = threshold_image(h, s, v)
    # cluster it
    clustered_mask, n_clusters = cluster_mask(mask, errosion_iterations, dilation_iterations)

    # get centers, assuming that the mask is fully inside the clustered mask
    centers, areas = get_center_and_areas(mask, clustered_mask, n_clusters)

    return centers, areas

def get_plants_from_rgb(img, time, errosion_iterations, dilation_iterations):
    """create plant objects from rgb image"""

    # determine centers and areas
    centers, areas = get_centers_and_areas_from_rgb(img, errosion_iterations, dilation_iterations)
    s = scale(*img.shape[:2])

    # create a new plant object for each found center
    plants = []
    for (center,area) in zip(centers,areas):
        if time is None:
            time = datetime.now()
        
        plant_obj = plant.Plant(time)
        
        plant_obj.set_origin(center[0], center[1])
        plant_obj.set_area_pxls(area,s)
        
        plants.append(plant_obj)

    return plants



def create_errosion_dilation_search(img):
    """Determine the number of plants for different combinations of errosion and dilation iterations.
    
    ____________________________________________________
    returns: 
    fig, results, a figure and a list of tuples containing the errosion and dilation iterations and the number of plants found
    """

    errosion_iterations = errosion_iterations_search_range
    dilation_iterations = dilation_iterations_search_range

    counts = np.zeros((errosion_iterations[1]-errosion_iterations[0], dilation_iterations[1]-dilation_iterations[0]))
    diff = np.zeros((errosion_iterations[1]-errosion_iterations[0], dilation_iterations[1]-dilation_iterations[0]))

    # loop through the combinations of errosion and dilation iterations
    for errosion in range(errosion_iterations[0], errosion_iterations[1]):
        for dilation in range(dilation_iterations[0], dilation_iterations[1]):
            plants = get_plants_from_rgb(img, None, errosion, dilation)#, errosion, dilation, threshold_h_lower, threshold_h_upper, threshold_s_lower, threshold_s_upper, threshold_v_lower, threshold_v_upper)
            counts[errosion-errosion_iterations[0], dilation-dilation_iterations[0]] = len(plants)  

    # calculate the difference to the mean of the neighbors
    for i in range(1, counts.shape[0]-1):
        for j in range(1, counts.shape[1]-1):
            # calculate the difference to the mean of the neighbors
            diff[i,j] = (counts[i,j]*4 - (counts[i-1,j] + counts[i+1,j] + counts[i,j-1] + counts[i,j+1])) 
            # weight it with the difference between all the neighbors
            diff[i,j] = diff[i,j] * (np.abs(counts[i-1,j] - counts[i+1,j]) + np.abs(counts[i,j-1] - counts[i,j+1])) 
            # make sure diff is an integer, round and convert to int
            diff[i,j] = round(diff[i,j])

    # make a plot 
    X, Y = np.meshgrid(np.arange(counts.shape[1]), np.arange(counts.shape[0]))

    #fig, ax = plt.subplots(2, 1  )
    #ax[0].imshow(counts, cmap='viridis', interpolation='nearest')
    Z = counts
    fig, ax = plt.subplots(2, 1)
    pcm = ax[0].pcolor(X, Y, Z,
                   cmap='PuBu_r', shading='auto')
    fig.colorbar(pcm, ax=ax[0], extend='max')
    ax[0].set_xlabel('dilation iterations')
    ax[0].set_ylabel('errosion iterations')

    ax[0].set_title('Number of plants')
    # write the found plants onto the image
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax[0].text(j, i, str(int(counts[i, j])), ha='center', va='center', color='black', fontsize=7)
    #colorbar
    # colorbar to axs[0]
    #cbar = plt.colorbar(ax[0].imshow(counts, cmap='viridis', interpolation='nearest'), ax=ax[0],
    #                    orientation='vertical', fraction=0.05)
    #cbar.set_label('number of plants')
    

    #ax[1].imshow(np.abs(diff), cmap='viridis', interpolation='nearest', norm = colors.LogNorm(vmin=0, vmax=np.max(np.abs(diff))))
    Z = np.abs(diff)

    pcm = ax[1].pcolor(X, Y, Z,
            norm=colors.LogNorm(vmin=1, vmax=Z.max()),
            cmap='PuBu_r', shading='auto')
    fig.colorbar(pcm, ax=ax[1], extend='max')

    ax[1].set_xlabel('dilation iterations')
    ax[1].set_ylabel('errosion iterations')
    
    ax[1].set_title('Cange metric')
    # colorbar to axs[1]
    #cbar = plt.colorbar(ax[1].imshow(np.abs(diff), cmap='viridis', interpolation='nearest'), ax=ax[1],
    #                    orientation='vertical', fraction=0.05, norm = colors.LogNorm(vmin=1, vmax=np.max(np.abs(diff))))
    #cbar.set_label('Change metric')

    # correct the ticks
    ax[0].set_xticks(np.arange(counts.shape[1]))
    ax[0].set_yticks(np.arange(counts.shape[0]))
    ax[1].set_xticks(np.arange(counts.shape[1]))
    ax[1].set_yticks(np.arange(counts.shape[0]))
    ax[0].set_xticklabels(np.arange(dilation_iterations[0], dilation_iterations[1]))
    ax[0].set_yticklabels(np.arange(errosion_iterations[0], errosion_iterations[1]))
    ax[1].set_xticklabels(np.arange(dilation_iterations[0], dilation_iterations[1]))
    ax[1].set_yticklabels(np.arange(errosion_iterations[0], errosion_iterations[1]))


    # find the minimas of the absolute difference, ignore the border
    minima = np.where(np.abs(diff[1:-1,1:-1]) == np.min(np.abs(diff[1:-1,1:-1])))

    # highlight the minima in the plot of the absolute difference
    results = []
    for i in range(len(minima[0])):
        ax[1].scatter(minima[1][i] + 1, minima[0][i] + 1, c='r', marker='x')
        print(f"found minimum at: {minima[0][i]}, {minima[1][i]}, with value: {counts[minima[0][i], minima[1][i]]}")
        # calculate corrosponding errosion and dilation and plants
        results.append((minima[0][i] +1+ errosion_iterations[0], minima[1][i]+1 + dilation_iterations[0], counts[minima[0][i], minima[1][i]]))
    plt.tight_layout()

    print(f"Minimum errosion: {errosion_iterations[0]}, Minimum dilation: {dilation_iterations[0]}")
    # print the minima
    print(f"found {len(minima[0])} minima at:\n")

    return fig, results




########################## Examples, tests, etc ##########################

def threshold_image_original(h, s, v):
    """
        Old version, used for validation atm
    """
    #mask = np.zeros(h.shape)
    mask = np.empty_like(h, dtype=bool)
    mask[(h > threshold_h_lower) & (h < threshold_h_upper) & (s > threshold_s_lower) & (s < threshold_s_upper) & (v > threshold_v_lower) & (v < threshold_v_upper)] = 1
    return mask
    
def use_example():
    img = plt.imread('capt0024.jpg')

    errosion_iterations = 3
    dilation_iterations = 13

    plants = get_plants_from_rgb(img, None, errosion_iterations, dilation_iterations)

    print(len(plants))

    
# write the areas onto the image centers
    fig, ax = plt.subplots(1, 1, figsize=(5, 5)  )
    ax.imshow(img, alpha= 0.7)

    # loop through the clusters and scatter an x at the center of each cluster
    for plant in plants:
        center = plant.get_origin()
        area = plant.get_area_pxls()
        age = plant.get_age()
        
        ax.scatter(center[0], center[1], c='r', marker='x')
        ax.text(center[0], center[1], str(area), c='black')
        
    fig.savefig("test_plants.png", dpi = 250)
    


def use_example_functions():
    img = plt.imread('/home/jonas/Desktop/projects/mip/testing/exampleImages/seedling2.jpg')

    errosion_iterations = errosion_iterations_default
    dilation_iterations = dilation_iterations_default



    # get the hsv channels
    h, s, v = get_hsv(img)
    # determine mask
    mask = threshold_image(h, s, v)
    # cluster it
    clustered_mask, n_clusters = cluster_mask(mask, errosion_iterations, dilation_iterations)

    # get centers, assuming that the mask is fully inside the clustered mask
    centers, areas = get_center_and_areas(mask, clustered_mask, n_clusters)

    print(f"found {len(centers)} plants:")

    # display the results,
    
    # write the areas onto the image centers
    fig, ax = plt.subplots(1, 1, figsize=(5, 5)  )
    ax.imshow(img, alpha= 0.8)

    # loop through the clusters and scatter an x at the center of each cluster
    for i in range(len(centers)):
        ax.scatter(centers[i][0], centers[i][1], c='r', marker='x')
        ax.text(centers[i][0], centers[i][1], str(areas[i]), c='b')

    fig.savefig("test.png", dpi = 250)

def compare_results(result1, result2):
    # Check if the shapes of the arrays are the same
    if result1.shape != result2.shape:
        print("The shapes of the arrays are different")
        return False

    # compare the elements of the arrays
    return np.all(result1 == result2)

def main():
    # Read the image
    image = plt.imread("capt0024.jpg")
    
    h, s, v = get_hsv(image)

    # Measure the runtime of the original thresholding function
    start_time = time.time()
    result1 = threshold_image_original(h, s, v)
    end_time = time.time()
    runtime1 = end_time - start_time
    
    # Measure the runtime of the test thresholding function
    start_time = time.time()
    result2 = threshold_image(h, s, v)
    end_time = time.time()
    runtime2 = end_time - start_time
    
    c = compare_results(result1, result2)
    print("The results are the same:", c)
    print("Runtime of threshold_image_old:", runtime1)
    print("Runtime of threshold_image_new:", runtime2)


if __name__ == "__main__":
   main()
