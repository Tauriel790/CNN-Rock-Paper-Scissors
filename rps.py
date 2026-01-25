# load necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from PIL import Image
import random, glob
from PIL import Image
from pathlib import Path


# --------------------------------------------------------------------- Loading the data ----------------------------------------------------------------------------------
# load the data into the working environment
data = tf.keras.utils.image_dataset_from_directory(
    "data",
    labels = "inferred",
    label_mode = "int",
    seed = 50,
    batch_size = 30,
    shuffle = True
)

# --------------------------------------------------------------- EDA (exploratory data analysis) ------------------------------------------------------------------------

# first of all the distribution of the images between the three classes "rock", "paper" and "scissors" is shown through the 
# use of a bar plot. As it can be seen from the resulting plot, the "scissors" class has the highest number of counts in respect
# to the other classes, even though all of the classes have similar distributions in the plot

directory = Path("data")
classes = ["paper", "rock", "scissors"]
counts = [len (os.listdir(os.path.join(directory, cls))) for cls in classes]
img_exts = (".jpg", ".jpeg", ".png")

# from the counts output, paper counts = 712, rock counts = 726, scissors counts = 750
print (counts)

# bar plot of the distribution of the variables 
plt.bar (classes, counts, color = ["red", "orange", "yellow"])
plt.title ("Number of images for each class")
plt.xlabel ("class")
plt.ylabel ("Count")
plt.show(block = False); plt.pause(2)

# now, a sample of 6 images of paper hand is shown 
plt.close("all")
paper = [p for p in (directory / "paper").iterdir()
         if p.is_file() and p.suffix.lower() in img_exts]
assert paper, "No images found in archive paper"

sample = random.sample(paper, k = min (6, len(paper)))

fig, axes = plt.subplots(1, 6, figsize = (20, 5))
for ax, img_path in zip (axes, sample):
    with Image.open(img_path) as im:
        im.thumbnail((256, 256))
        ax.imshow(im)
        ax.axis("off")

plt.suptitle("Random paper's images examples", y = 2)
plt.subplots_adjust(wspace = 1)
plt.tight_layout()
plt.show(block = False); plt.pause(2)

# we do the same also for rock hands
plt.close("all")
rock = [p for p in (directory / "rock").iterdir()
         if p.is_file() and p.suffix.lower() in img_exts]
assert rock, "No images found in archive rock"

sample = random.sample(rock, k = min (6, len(rock)))

fig, axes = plt.subplots(1, 6, figsize = (20, 5))
for ax, img_path in zip (axes, sample):
    with Image.open(img_path) as im:
        im.thumbnail((256, 256))
        ax.imshow(im)
        ax.axis("off")

plt.suptitle("Random rock's images examples", y = 2)
plt.subplots_adjust(wspace = 1)
plt.tight_layout()
plt.show(block = False); plt.pause(2)

# then also for the scissors hands
plt.close("all")
scissors = [p for p in (directory / "scissors").iterdir()
         if p.is_file() and p.suffix.lower() in img_exts]
assert scissors, "No images found in archive scissors"

sample = random.sample(scissors, k = min (6, len(scissors)))

fig, axes = plt.subplots(1, 6, figsize = (20, 5))
for ax, img_path in zip (axes, sample):
    with Image.open(img_path) as im:
        im.thumbnail((256, 256))
        ax.imshow(im)
        ax.axis("off")

plt.suptitle("Random scissors' images examples", y = 2)
plt.subplots_adjust(wspace = 1)
plt.tight_layout()
plt.show(block = False); plt.pause(2)

# Before processing it's important to check the characteristics of the images we have in the three classes that are present
# in the dataset we are analyzing. In particular, we want to check:

# 1) the dimensions of the images among the three classes 

for cls in classes: 
    images = [p for p in (directory / cls).iterdir()
              if p.is_file() and p.suffix.lower() in img_exts]
    
    widths, heights, channels = [], [], []
    for img_path in images:
        with Image.open(img_path) as im:
            widths.append(im.width)
            heights.append(im.height)
            channels.append(len(im.getbands()))

    print(f"\nWidth: min = {min(widths)}, max = {max(widths)}, average = {np.mean(widths)}")
    print(f"Height: min = {min(heights)}, max = {max(heights)}, average = {np.mean(heights)}")
    print(f"Channels: {set(channels)}")

# 2) Pixel intensity analysis: This shows if the images are well exposed or we have to adjust them
# because they are too dark or too bright. We will plot the pixel intensity distribution for a sample of images from each class.

# - for Paper class
plt.close("all")
sample_images = random.sample(list((directory / "paper").iterdir()), 10)

intensities = []
for img_path in sample_images:
    img_array = np.array(Image.open(img_path))
    intensities.extend(img_array.flatten())

plt.figure(figsize = (10, 4))
plt.hist(intensities, bins = 50, color = 'blue', alpha = 0.7)
plt.title ("Pixel Intensity Distribution for Sample Paper Images")
plt.xlabel ("Pixel Intensity")
plt.ylabel ("Frequency")
plt.show(block = False); plt.pause (2)

# - for Rock class
plt.close("all")
sample_images = random.sample(list((directory / "rock").iterdir()), 10)

intensities = []
for img_path in sample_images:
    img_array = np.array(Image.open(img_path))
    intensities.extend(img_array.flatten())

plt.figure(figsize = (10, 4))
plt.hist(intensities, bins = 50, color = 'pink', alpha = 0.7)
plt.title ("Pixel Intensity Distribution for Sample Rock Images")
plt.xlabel ("Pixel Intensity")
plt.ylabel ("Frequency")
plt.show(block = False); plt.pause (2)

# - for Scissors class
plt.close("all")
sample_images = random.sample(list((directory / "scissors").iterdir()), 10)

intensities = []
for img_path in sample_images:
    img_array = np.array(Image.open(img_path))
    intensities.extend(img_array.flatten())

plt.figure(figsize = (10, 4))
plt.hist(intensities, bins = 50, color = 'violet', alpha = 0.7)
plt.title ("Pixel Intensity Distribution for Sample Scissors Images")
plt.xlabel ("Pixel Intensity")
plt.ylabel ("Frequency")
plt.show(block = False); plt.pause (2)

# 3) Visual check of the quality of the images

# for Paper class
sample_img_path = random.choice(list((directory / "paper").iterdir()))
with Image.open(sample_img_path) as img:
    img_array = np.array(img)

    print(f"Example of image properties for Paper class:")
    print(f" Shape: {img_array.shape}")

# for Rock class
sample_img_path = random.choice(list((directory / "rock").iterdir()))
with Image.open(sample_img_path) as img:
    img_array = np.array(img)

    print(f"Example of image properties for Rock class:")
    print(f" Shape: {img_array.shape}")
    print(f" Data Type: {img_array.dtype}")
    print(f" Value Range: [{img_array.min()}, {img_array.max()}]")

# for Scissors class
sample_img_path = random.choice(list((directory / "scissors").iterdir()))
with Image.open(sample_img_path) as img:
    img_array = np.array(img)

    print(f"Example of image properties for Scissors class:")
    print(f" Shape: {img_array.shape}")
    print(f" Data Type: {img_array.dtype}")
    print(f" Value Range: [{img_array.min()}, {img_array.max()}]")

# 4) Brightness and Contrast Analysis
print ("\n--- Brightness and Contrast Analysis: ---")

brightness_data = []
contrast_data = []

for cls in classes:
    sample_images = random.sample(list((directory / cls).iterdir()), 50)

    brightness_values =[]
    contrast_values = []

    for img_path in sample_images:
        img_array = np.array(Image.open(img_path)) # (brightness = mean of all pixel values)
        brightness_values.append(np.mean(img_array))

        contrast_values.append(np.std(img_array)) # (contrast = std deviation of pixel values)

    avg_brightness = np.mean(brightness_values)
    avg_contrast = np.mean(contrast_values)

    brightness_data.append(avg_brightness)
    contrast_data.append(avg_contrast)

    print(f"\n{cls.upper()}:")
    print(f" Average Brightness: {avg_brightness} (range: 0-255)")
    print(f" Average Contrast: {avg_contrast}")

# Now a visualization of the brightness and constrast values will be made for comparison
plt.close("all")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

# Brightness comparison plot
ax1.bar(classes, brightness_data, color = ["red", "orange", "yellow"])
ax1.set_title("Average Brightness by Class")
ax1.set_xlabel("Class")
ax1.set_ylabel("Average Brightness by class)")
ax1.set_ylim([0, 255])

# Contrast comparison plot
ax2.bar(classes, contrast_data, color = ['purple', "violet", "pink"])
ax2.set_title("Average Contrast by Class")
ax2.set_xlabel("Class")
ax2.set_ylabel("Contrast (Std Dev of Pixel Values)")

plt.tight_layout()
plt.show(block = False); plt.pause (2)

# 5) Brightness and Contrast Analysis scatter plot
plt.close("all")
fig, ax = plt.subplots(figsize = (8, 6))

colors_map = {"paper": "blue", "rock": "violet", "scissors": "pink"}

for cls in classes:
    sample_images = random.sample(list((directory / cls).iterdir()), 30)
    brightness_values = []
    contrast_values = []

    for img_path in sample_images:
        img_array = np.array(Image.open(img_path))
        brightness_values.append(np.mean(img_array))
        contrast_values.append(np.std(img_array))

    ax.scatter(brightness_values, contrast_values, alpha = 0.6, s = 50, c = colors_map[cls], label = cls.capitalize())

ax.set_title("Brightness vs Contrast Scatter Plot")
ax.set_xlabel("Brightness (Mean Pixel Value)")
ax.set_ylabel("Contrast (Std Dev of Pixel Values)")
ax.legend()
ax.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show(block = False); plt.pause (2)

# Now we will create a summary table of all the findings obtained till now
print(("\n ------ DATASET SUMMARY ------"))
print(f"Total number of images: {sum(counts)}")
print(f"Number of classes: {len(classes)}")
print(f"Class Distribution: {dict(zip(classes, counts))}")
print("Class balance: ", "Balanced" if max(counts) - min(counts) < 0.1 * sum(counts) else "Imbalanced")


