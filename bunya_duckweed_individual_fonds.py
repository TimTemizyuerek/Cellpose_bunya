## import packages and functions
import torch
import os
import numpy as numpy
import cv2
import fnmatch
import pandas as pandas
import glob
from cellpose import models, io, train

## load dir
pic_dir = os.environ["TMPDIR"] + "/analysis_images/"
out_dir = os.environ["TMPDIR"] + "/analysis_output/"
os.makedirs(out_dir, exist_ok=True)

## area of square in mm^2
square_area = 100
## removes all ROIs with fewer pixels
min_pixel_filter = 30

## load trained model model
model = models.CellposeModel(gpu=True, pretrained_model="/scratch/user/uqttemiz/trained_models/standard_model_31_training_img")

## create lists
results = []
runtimes = []

## create iterator
files = sorted(f for f in os.listdir(pic_dir)
               if not f.startswith(".") and f.lower().endswith((".png")))

## loop through files
for n in range(len(files)):
    
    ## LOAD FILE
    runner_pic_name = fnmatch.filter(os.listdir(pic_dir), '*.png*')[n]
    runner_file_location = (pic_dir + "/" + runner_pic_name)
    ## save basename for output
    runner_basename = runner_pic_name.replace(".png","")
    runner_pic = cv2.imread(runner_file_location)
    if runner_pic is None:
        print(f"Could not read {runner_pic_name}, skipping", flush=True)
        continue    

    ## FIND RED SQUARE AND GENERATE NUMBER TO RELATE PIXELS TO MM^2
    ## convert to HSV (Hue, Saturation, Value) which makes identification based on colour easier
    hsv = cv2.cvtColor(runner_pic, cv2.COLOR_BGR2HSV)
    ## define the red colour to search for
    lower_red1, upper_red1 = numpy.array([0,70,50]), numpy.array([10,255,255])
    lower_red2, upper_red2 = numpy.array([170,70,50]), numpy.array([180,255,255])
    ## identify red in picture
    mask = cv2.bitwise_or(cv2.inRange(hsv,lower_red1,upper_red1),
                        cv2.inRange(hsv,lower_red2,upper_red2))
    ## find largest outline of a continuous red area (so that I don't have to worry about the two squares)
    outlines = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not outlines:
    	print(f"No red square found in {runner_pic_name}, skipping")
    	continue
    largest_outline = max(outlines, key=cv2.contourArea)    

## count pixels in largest outline
    outline_pixels = cv2.contourArea(largest_outline)  # returns area in pixels²
    ## scale with real square area in mm (i.e. 100)
    mm2_per_pixel = square_area / outline_pixels
    
    ## RUNN CELLPOSE MODEL
    runner_pic_rgb = cv2.cvtColor(runner_pic, cv2.COLOR_BGR2RGB)	    
    mask = model.eval(runner_pic_rgb)[0]
    
    ## ANALYSE INDIVIDUAL FRONDS
    ## get unique Regions Of Interest (ROI)
    roi_labels = numpy.unique(mask)
    ## check pixel number of each ROI
    roi_pixels = []
    for m in roi_labels:
        ## count pixels
        pixel_count = numpy.sum(mask == m)
        ## save if ROI is bigger than min_pixel_filter (set above)
        if pixel_count > min_pixel_filter:
            runner_area_mm2 = mm2_per_pixel * pixel_count
            roi_pixels.append((m, runner_area_mm2))
    ## make readable and export to csv
    roi_mm2 = pandas.DataFrame(roi_pixels, columns=['labels', 'mm2'])
    roi_mm2.to_csv(os.path.join(out_dir, f"{runner_basename}.csv"), index=False)

    ## CREATE OVERLAY FOR MANUAL QUALITY CONTROL
    ## copy image, and get all ROIs (removing background)
    empty_overlay = runner_pic.copy()
    roi_labels_zero_removed = roi_labels[roi_labels > 0]
    ## generate random colours for the ROIs
    rng = numpy.random.default_rng(1)
    colors = rng.integers(0, 255, size=(len(roi_labels_zero_removed), 3))
    ## cover each ROI with transparent colour
    for m, col in zip(roi_labels_zero_removed, colors):
        empty_overlay[mask == m] = (0.5 * empty_overlay[mask == m] + 0.5 * col).astype(numpy.uint8)

    ## compute ROI outlines
    edges = numpy.zeros(mask.shape, numpy.uint8)
    for m in roi_labels_zero_removed:
        obj_mask = (mask == m).astype(numpy.uint8) * 255
        edges = cv2.Canny(obj_mask, 50, 150)
        edges_dilated = cv2.dilate(edges, numpy.ones((3,3), numpy.uint8))
        edges = cv2.bitwise_or(edges, edges_dilated)
        ## draw outlines
        empty_overlay[edges > 0] = (0, 0, 255)
    ## export overlay with basename
    overlay_path = os.path.join(out_dir, f"{runner_basename}_overlay.png")
    cv2.imwrite(overlay_path, empty_overlay)
