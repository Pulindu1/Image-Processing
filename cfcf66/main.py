"""
NOTE: 
RUN PROGRAM:
Run:
    python main.py image_processing_files/xray_images/
Test:
    python classify.py --data=Results --model=classifier.model
"""

import cv2
import os
import numpy as np

# find the images
base_directory = os.path.dirname(os.path.abspath(__file__))
source_directory = os.path.join(base_directory, 'xray_images')
destination_directory = os.path.join(base_directory, 'Results')

# create new dir
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

################## Functions #####################
##################################################

# 'rotate' images
def apply_perspective_transformation(img, width=256, height=256):
    input_points = np.float32([[9, 16], [233, 6], [30, 236], [251, 228]])
    new_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # calculate and apply transformation matrix
    matrix = cv2.getPerspectiveTransform(input_points, new_points)
    transformed_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR) 
    return transformed_img

def bilateral(img, d=10, sigmaColour=15, sigmaSpace=15):
    return cv2.bilateralFilter(img, d, sigmaColour, sigmaSpace)

def non_local_means_denoising(img, h=12, hForColourComponents=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(img, None, h, hForColourComponents, templateWindowSize, searchWindowSize)

def unsharp_mask(image, sigma=1.0, strength=2.5):

    # Blur image
    blurredImg = cv2.GaussianBlur(image, (0, 0), sigma)

    sharpenedImg = cv2.addWeighted(image, 1.0 + strength, blurredImg, -strength, 0)
    return sharpenedImg

def colour_imbalance(img, hue_shift=2, saturation_increase=100, value_decrease=80, background_hue_threshold=(100, 140), background_value_threshold=180):
    # Convert to HSV
    hsv_ColourSpace = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_ColourSpace)
    
    # Darken background
    dark_background_mask = np.logical_and.reduce((h >= background_hue_threshold[0], h <= background_hue_threshold[1], v >= background_value_threshold))
    v[dark_background_mask] -= value_decrease
    v[v < 0] = 0  #check for negative vals
    h = np.mod(h + hue_shift, 180)
    s = cv2.add(s, saturation_increase)
    
    # Merge
    hsv_adjusted = cv2.merge([h, s, v])
    img_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return img_adjusted

def inpainting(post_warp):

    # Convert to greyscale
    grey = cv2.cvtColor(post_warp, cv2.COLOR_BGR2GRAY)

    # apply binary thresholding
    threshold_val, threshold_img = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_not(threshold_img)
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(mask,kernel,iterations = 1)

    ns = cv2.inpaint(post_warp, dilate, 3, cv2.INPAINT_NS)
    return ns

def adjust_brightness(img, brightness_increase=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    h, s, v = cv2.split(hsv)

    v = cv2.add(v, brightness_increase)         # increase brightness
    v[v > 255] = 255                            # cap the value
    hsv2 = cv2.merge((h, s, v))            # Merge channels
    brightened_img = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)  # back to BGR
    return brightened_img

def median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

def adjust_contrast(image, clip_limit=2.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(11, 11))
    cl_applied = clahe.apply(l)
    
    # Merge 
    lab = cv2.merge((cl_applied, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


################ End of Functions ################
##################################################

# apply the enhancements
def process_images(image_path, destination_path, median_kernel_size = 5):
    img = cv2.imread(image_path)
    
    # fill hole
    img = inpainting(img)

    # fix noise
    img = bilateral(img)
    img = median_filter(img, kernel_size=median_kernel_size)
    img = non_local_means_denoising(img)

    # fix warping
    img = apply_perspective_transformation(img)

    # sharpen
    img = unsharp_mask(img)

    # adjust colours inbalance
    img = colour_imbalance(img)

    # contrast and brightness
    img = adjust_contrast(img)
    img = adjust_brightness(img)

    cv2.imwrite(destination_path, img)

print("Processing images...")
for filename in os.listdir(source_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(destination_directory, filename)
        process_images(source_path, destination_path)

print(f"COMPLETED! Processed images saved at  {destination_directory}")
