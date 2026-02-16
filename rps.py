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
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import itertools

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
plt.hist(intensities, bins = 50, color = 'green', alpha = 0.7)
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
plt.hist(intensities, bins = 50, color = 'purple', alpha = 0.7)
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

# 6) RGB Channel Analysis (color distribution patterns of the images present in out dataset)
plt.close("all")
fig, axes = plt.subplots(1, 3, figsize = (18, 5))

for idx, cls in enumerate(classes):
    sample_images = random.sample(list((directory / cls).iterdir()), 20)

    r_means, g_means, b_means = [], [], []
    for img_path in sample_images:
        img_array = np.array(Image.open(img_path))
        r_means.append(np.mean(img_array[:, :, 0]))
        g_means.append(np.mean(img_array[:, :, 1]))
        b_means.append(np.mean(img_array[:, :, 2]))

    # Bar plot of average channel values
    channels = ["blue", "green", "yellow"]
    values = [np.mean(r_means), np.mean(g_means), np.mean(b_means)]

    axes[idx].bar(channels, values, color = ["blue", "green", "yellow"], alpha = 0.7)
    axes[idx].set_title(f"Average RGB Channel Values {cls.capitalize()}")
    axes[idx].set_ylabel("Average Pixel Value")
    axes[idx].set_ylim([0, 200])

plt.tight_layout()  
plt.show(block = False); plt.pause(2)

# SUMMARY OF THE FINDINGS OBTAINED FROM THE EDA

print("\n ------ EDA FINDINGS SUMMARY ------")

# Basic dataset information
print(f"Total number o images: {sum(counts)}")
print(f"Number of classes: {len(classes)}")
print(f"Class Distribution: {dict(zip(classes, counts))}")
print(f"Class balance: ", "Balanced" if max(counts) - min(counts) < 0.1 * sum(counts) else "Imbalanced")

# Image characteristics and properties 
sample_img_path = random.choice(list((directory / classes[0]).iterdir()))
with Image.open(sample_img_path) as img:
    sample_array = np.array(img)
    img_width = img.width
    img_height = img.height
    img_channels = sample_array.shape[2] if len(sample_array.shape) == 3 else 1
    img_dtype = sample_array.dtype
    img_min = sample_array.min()
    img_max = sample_array.max()

print(f"Image dimension: {img_width} x {img_height} pixels ")
print(f"Color channels: {img_channels} (RGB)")
print(f"Data type: {img_dtype}")
print(f"Pixel Value Range: [{img_min}, {img_max}]")

# Statistical findings
print(f"\nAverage Brightness by Class:")
for cls, brightness in zip(classes, brightness_data):
    print(f" {cls.capitalize()}: {brightness.round(2)}")

print(f"\nAverage Contrast by Class:")
for cls, contrast in zip(classes, contrast_data):
    print(f"{cls.capitalize()}: {contrast.round(2)}")

# Key onservations
print("\nKEY OBSERVATIONS:")
print("- The dataset is relatively balanced across the three classes with minimal class imbalance")
print("- All the images have consistent dimensions")
print("- Images are well-exposed with average brightness levels within acceptable ranges")
print("- The green channel dominates due to the green screen background that is present in all the inages among the three classes")
print("All three classes sho similar RGB distribution patterns")
print("No significant color bias detected between the classes")

# ------------------------------------------------------------- TRAIN, VALIDATION AND TEST SETS SPLIT  ------------------------------------------------------------------------

# First, it is important to split the data into train and test set before any processing to avoid data leakage. This ensures that the test set data remain unseen during
# the training process and that no information from the test set influences the model training. Best practice: splitting at File Level before building any tf.data pipelines.
# This guarantees:
# - zero leakage
# - Full reproducibility across sessions and environments
# - Transparent, auditable splits (so we can inspect exactly which files are where)

# Collecting all the image paths and their labels from the directory
all_images_paths = []
all_labels = []

for label_idx, cls in enumerate(classes):
    cls_paths = [
        str(p) for p in (directory / cls).iterdir()
        if p.is_file() and p.suffix.lower() in img_exts
    ]
    all_images_paths.extend(cls_paths)
    all_labels.extend([label_idx] * len(cls_paths))

# Shuffling at the file level with a fixed seed for full reproducibility
combined = list(zip(all_images_paths, all_labels))
random.seed(42)
random.shuffle(combined)
all_images_paths, all_labels = zip(*combined)
all_images_paths = list(all_images_paths)
all_labels = list(all_labels)

# Computing the split indices (70% train / 15% validation / 15% test)
total = len(all_images_paths)
train_end = int(0.70 * total)
val_end = int(0.85 * total)

train_paths, train_labels = all_images_paths[:train_end], all_labels[:train_end]
val_paths, val_labels = all_images_paths[train_end:val_end], all_labels[train_end:val_end]
test_paths, test_labels = all_images_paths[val_end:], all_labels[val_end:]

print(f"\nFile-level split:")
print(f"Train: {len(train_paths)} images")
print(f"Validation: {len(val_paths)} images")
print(f"Test: {len(test_paths)} images")
print(f"Total: {len(train_paths) + len(val_paths) + len(test_paths)} (should equal {total})")

# Verifying now that there is zero overlap between the splits done
train_set = set(train_paths)
val_set = set(val_paths)
test_set = set(test_paths)

assert len (train_set & val_set) == 0, "Leakage: train and val share images!"
assert len (train_set & test_set) == 0, "Leakage: train and test share images!"
assert len (val_set & test_set) == 0, "Leakage: val and test share images!"
print("Zero overlap confirmed between all splits.")

# Then, we verify the class distribution across the splits
print("Class distribution:")
for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
    vals, cnts = np.unique(split_labels, return_counts = True)
    dist = {classes[v]: c for v, c in zip(vals, cnts)}
    print(f"{split_name}: {dist}")

# Now, tf.data pipelines are built from the file lists. Images are loaded and resized in this process

def load_and_resize (path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels = 3, expand_animations = False)
    image = tf.image.resize(image, [150, 150])
    image.set_shape([150, 150, 3])
    return image, label

BATCH_SIZE = 30

train_data = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(load_and_resize, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)

val_data = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(load_and_resize, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)

test_data = (
    tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    .map(load_and_resize, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)

print(f"Train batches: {tf.data.experimental.cardinality(train_data).numpy()}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_data).numpy()}")
print(f"Test batches: {tf.data.experimental.cardinality(test_data).numpy()}")

# ---------------------------------------------------------------------- PROCESSING OF THE DATA ------------------------------------------------------------------------------
# Now that the data has been splitted into train, validation and test sets, we can proceed with the processing of the data. Since the dimensions of the images have already 
# been set to 150 x 150 pixels during the loading phase (image resizing step), we can now focus on normalizing the pixel values to a range of [0, 1] and applying data 
# augmentation techniques to enhance the diversity of the training dataset.

# 1) NORMALIZATION: -------------------------------------------------------------------------------------------------------------------------------------------------------
# First of all, we normalize the pixel values to a range of [0, 1] by rescaling them by a factor of 1./255
# This divides all the pixel values by 255, effectively transforming the original range of [0, 255] to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply the normalization to all the three sets 
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# Now we verify that the normalization has been applied correctly by checking the pixel value ranges in a sample batch
for images, labels in train_data.take(1):
    print(f"\nOriginal image shape: {images.shape}")
    print(f"Pixel value range after normalization: [{tf.reduce_min(images).numpy()}, {tf.reduce_max(images).numpy()}]")
    print(f"Expected range: [0.0, 1.0]")

    if tf.reduce_min(images) >= 0.0 and tf.reduce_max(images) <= 1.0:
        print("Normalization applied correctly.")
    else:
        print("Normalization not applied correctly.")

# 2) DATA AUGMENTATION: ----------------------------------------------------------------------------------------------------------------------------------------------------
# Data augmentation is a technique used to artificially increase the size and diversity of the training dataset by applying random transformations to the images. 
# This helps improve the model's ability to generalize to unseen data and reduces overfitting. The augmentation techniques will be applied only to the training set, 
# not to the validation or test sets because they should contain unaltered data to accurately evaluate the model's performance.

# Based on the characteristics of the rock, paper, scissorsdataset, the data augmentation techniques that will be applied include:
# 1) Random horizontal flipping: This flips the image horizontally with a 50% chance, which helps the model learn to recognize objects from different orientations.
# 2) Random rotation: This rotates the image randomly by +- 72 degrees (0.2 x 360°), which helps the model become invariant to natural hand tilting and different angles.
# 3) Random zooming: This randomly zooms in or out by +- 10% (range: 90% to 110%), which helps the model learn to recognize gestures at different scales and distances from the camera.
# 4) Random translation: This shifts the image randomly along the width and height within a specified range (e.g., ±20%), which helps the model become invariant to small
#  translations of the objects in the images.

# Vertical flipping and color augmentation were not included because they could distort the natural appearance of the hand gestures in the images and potentially confuse the model.

# Define the data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.2, 0.2)
])

# Now we apply the data augmentation techniques only to the training set
# The training = True parameter ensures that the augmentation is only active during training and not during evaluation
train_data = train_data.map(
    lambda x, y: (data_augmentation(x, training = True), y),
    num_parallel_calls = tf.data.AUTOTUNE
)

# 3) Pipeline optimization: ------------------------------------------------------------------------------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

# Adding prefetching to all datasets for improved performance
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# 4) VISUALIZATION OF AUGMENTED IMAGES: ---------------------------------------------------------------------------------------------------------------------------------
# Now let's visualize some augmented images to see the effects that the data augmentation techniques had on the original images
plt.close("all")

fig, axes =plt.subplots(3, 5, figsize = (18, 10))

for class_idx, cls in enumerate(classes):
    # get one image from this class
    class_batch = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels = "inferred",
        label_mode = "int",
        class_names = classes,
        batch_size = 1,
        image_size = (150, 150),
        shuffle = True,
        seed = 42 + class_idx
    )

    for images, labels in class_batch.take(1):
        original_image = normalization_layer(images)

        # Column 0: Original (normalized but not augmented)
        axes[class_idx, 0].imshow(original_image[0])
        axes[class_idx, 0].set_title(f"{cls.capitalize()} \nOriginal", fontsize = 12, fontweight = "bold")

        axes[class_idx, 0].axis("off")

        # Set border for original image
        for spine in axes[class_idx, 0].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(2)

        # Column 1-4: Augmented versions of the images
        for aug_idx in range(1, 5):
            augmented_image = data_augmentation(original_image, training = True)
            axes[class_idx, aug_idx].imshow(augmented_image[0])
            axes[class_idx, aug_idx].set_title(f"Augmented {aug_idx}", fontsize = 12, fontweight = "bold")
            axes[class_idx, aug_idx].axis("off")

plt.suptitle("Data Augmentation Examples: Original vs Augmented", fontsize = 16, fontweight = "bold", y = 1.05)
plt.tight_layout()
plt.show(block = False); plt.pause (3)

# ------------------------------------------------------------------- CNN ARCHITECTURES  ----------------------------------------------------------------------------------
# Now that the data has been properly processed, we can proceed with the definition of the CNN architectures that will be used for the classification task. Three different CNN architectures
# will be defined and compared: a simple CNN, a deeper CNN and a transfer learning model using a pre-trained network (MobileNetV2). Each architecture will be built, compiled, trained and evaluated
# separately to assess their performance on the rock, paper, scissors dataset.

# ------------------------------------------------------------- 1) SIMPLE/BASELINE CNN ARCHITECTURE: -----------------------------------------------------------------------
# This baseline model employs minimal depth to test whether basi feature extraction is sufficient for the 3 class rock-paper-scissors task. With only 2 convultional block and no dropout,
# this model serves as a benchmark for more complex architectures.

model_1_baseline = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (150, 150, 3), name = "input"),

    # Convolutional Block 1
    # - 32 filters
    # - 3x3 kernel size (which is a standard choice for capturing local patterns in images
    # - ReLU activation function to introduce non-linearity
    tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", name = "conv1"),
    # MaxPooling layer to reduce spatial dimensions and retain important features
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool1"),

    # Conolutional Block 2
    # - 64 filters: Learn more complex features
    # - 3x3 kernel size
    # - Progressively increasing the number of filters helps the model capture a wider range of features at different levels of abstraction
    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu", name = "conv2"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool2"),

    # Classification head
    # Flatten layer to convert 2D feature maps to 1D feature vectors
    tf.keras.layers.Flatten(name = "flatten"),
    # Dense layer with 64 units and ReLU activation to learn complex patterns
    tf.keras.layers.Dense(64, activation = "relu", name = "dense1"),
    # Output layer with 3 units (one for each class) and softmax activation for multi-class classification
    tf.keras.layers.Dense(3, activation = "softmax", name = "output")
], name = "Baseline_CNN")

# Summary of the model architecture
print ("\nModel architecture - Simple/Baseline CNN:")
model_1_baseline.summary()

# ------------------------------------------------------------ 2) INTERMIDIATE CNN ARCHITECTURE: ------------------------------------------------------------------------------
# This model compared with the baseline CNN architecture implementd above, adds a third convolutional block (with 128 filters) to capture more complex feature hierarchies.
# Additionally, dropout layers (dropout rate of 0.5) are introduced after each convolutional block to mitigate overfitting by randomly deactivating neurons during training. 
# This helps the model generalize better to unseen data.

model_2_intermidiate = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape = (150, 150, 3), name = "input"),

    # Convolutional Block 1:
    # - 32 filters which will learn basic features such as edges, colors and simple textures
    # - 3x3 kernel size to capture local patterns in the images
    # - ReLU activation which intriduces non-linearity to help the model learn complex relationships between features
    tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", name = "conv1"),
    # MaxPooling layer to reduce the spacial dimensions of the faature maps and retain the most important features
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool1"),

    # Convolutional Block 2:
    # - 64 filters to learn more complex features such as combinations of edges and textures like hand shapes and so on ..
    # - The filters where doubled compared to the first block to allow the model to capture a wider range of features at different levels of abstraction
    # - 3x3 kernel size to continue capturing local patterns
    # - ReLU activation to maintain non-linearity in the model
    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu", name = "conv2"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool2"),

    # Convolutional Block 3:
    # - 128 filters to learn even more complex features and higher-level representations of the images, such as specific hand gestures and finer details
    # - 3x3 kernel size to maintain consistency in capturing local patterns
    # - ReLU activation to continue introducing non-linearity
    tf.keras.layers.Conv2D(128, (3, 3), activation = "relu", name = "conv3"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool3"),

    # Classification head:
    # Flatten layer to convert the 2D feature maps into a 1D feature vector that can be fed into the dense layers
    tf.keras.layers.Flatten (name = "flatten"),
    
    # Dense layer with 128 units and ReLU activation to learn complex patterns and relationships between the features extracted by the convolutional blocks
    tf.keras.layers.Dense (128, activation = "relu", name = "dense1"),

    # Dropout layer with a dropout rate of 0.5 to mitigate overfitting by randomly deactivating neurons during training. This helps the model to generalize 
    # better to unseen data
    tf.keras.layers.Dropout(0.5, name = "dropout1"),

    # Output layer with 3 units (one for each class) and softmax activation for multi-class classification
    tf.keras.layers.Dense (3, activation = "softmax", name = "output")
], name = "Intermediate_CNN")

# Summary of the model architecture
model_2_intermidiate.summary()

# ------------------------------------------------------------ 3) ADVANCED CNN ARCHITECTURE ------------------------------------------------------------------------------------
# This model implements a modern CNN architecture following best practices from state of the art networks like ResNet and EfficientNet. The architecture features 4 convolutional 
# blocks with progressive filter expansion (32-64-128-256) and incorporates BatchNormalization for training stabilization.

model_3_advanced = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape = (150, 150, 3), name = "input"),

    # ---------------------- Convolutional Block 1 (basic feature extraction): --------------------------------
    # - first convolutional layer with 32 filters 
    # - 3x3 kernel size 
    # - use_bias = False because BatchNormalization provides its own bias term
    tf.keras.layers.Conv2D(32, 3, padding = "same", use_bias = False, name = "conv1"),

    # BatchNomralization normalizes the output of the convolution to have mean = 0 and variance = 1
    # This stabilizes training and allows higher learning rates
    tf.keras.layers.BatchNormalization(name = "bn1"),

    # ReLU activation applied after normalization
    tf.keras.layers.Activation("relu", name = "relu1"),

    # MaxPooling reduces spatial dimensions by half
    tf.keras.layers.MaxPooling2D(name = "maxpool1"),

    # ---------------------- Convolutional Block 2 (Mid level featue extraction): --------------------------------
    # - 64 filters to learn more complex feature combinations
    tf.keras.layers.Conv2D(64, 3, padding = "same", use_bias = False, name = "conv2"),
    tf.keras.layers.BatchNormalization(name = "bn2"),
    tf.keras.layers.Activation("relu", name = "relu2"),

    # Spatial dimension reduction
    tf.keras.layers.MaxPooling2D(name = "maxpool2"),

    # ---------------------- Convolutional Block 3 (high level feature extraction): --------------------------------
    # 128 filters to capture hand gestures shaoes and specific patterns
    tf.keras.layers.Conv2D(128, 3, padding = "same", use_bias = False, name = "conv3"),
    tf.keras.layers.BatchNormalization(name = "bn3"),
    tf.keras.layers.Activation("relu", name = "relu3"),
    tf.keras.layers.MaxPooling2D (name = "maxpool3"),

    # ---------------------- Convolutional Block 4 (Abstract feature extractions): --------------------------------
    # 256 filters to learn the most abstract representation of complete gestures
    tf.keras.layers.Conv2D(256, 3, padding = "same", use_bias = False, name = "conv4"),
    tf.keras.layers.BatchNormalization(name = "bn4"),
    tf.keras.layers.Activation("relu", name = "relu4"),
    tf.keras.layers.MaxPooling2D(name = "maxpool4"),

    # Dropout layer with rate 0.3 to prevent overfitting by randomly deactivating 30% of neurons during training
    tf.keras.layers.Dropout(0.3, name = "dropout1"),

    # ---------------------- Global Average Pooling: Dimensionality reduction ----------------------------------------------
    # Global average pooling averages each 9x9 feature map into a single value.
    # This is a more modern alrernative of flatten that reduces overfitting and improves generalization
    tf.keras.layers.GlobalAveragePooling2D(name = "global_avg_pool"),

    # ---------------------- Classification head: ----------------------------------------------------------------------
    # Dense layer with 128 units to learn complex decision boundaries from the 256 global features
    tf.keras.layers.Dense (128, activation = "relu", name = "dense1"),

    # Dropout layer to further prevent overfitting in the classification head
    tf.keras.layers.Dropout(0.3, name = "dropout2"),

    # Output layer with 3 units (one pr class: paper, rock, scissors) and softmax activatio.
    # Softmax converts raw scores into probability distribution summing to 1.0
    tf.keras.layers.Dense(3, activation = "softmax", name = "output")
], name = "Advanced_CNN")

# Summary of the model architecture
model_3_advanced.summary()

# ------------------------------------------------------------ COMPILING THE 3 MODELS ----------------------------------------------------------------------

# 1) COMPILING THE BASELINE CNN MODEL --------------------------------------------------------------------------------------
print ("\nCompiling the Baseline CNN model...")

model_1_baseline.compile(
    # The Adam optimizer is chosen for its efficiency and adaptive learning rate capabilities, which can help the model converge faster and achieve better 
    # performance on the rock-paper-scissors classification task. And we use Adam with the defaul learning rate of 0.001 which is a common starting point 
    # for many tasks and often works well without the need for extensive hyperparameter tuning.
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    
    # Loss function: Sparse Categorical Crossentropy is used for multi-class classification problems where the labels are provided as integers. It is 
    # suitable for our rock-paper-scissors task because we have three classes (rock, paper, scissors) and the labels are encoded as integers (0, 1, 2). 
    # This loss function computes the cross-entropy loss between the true labels and the predicted probabilities output by the model, which helps to 
    # optimize the model's performance in classifying the images correctly.
    # - "sparse" => indicates that the labels are provided as integers rather than one-hot encoded vectors
    # - "Categorical" => indicates that it is a multi-class classification problem (paper, rock, scissors)
    # - "Crossentropy" => measures the difference between the true labels and the predicted probabilities, which is what we want to minimize during training
    loss = "sparse_categorical_crossentropy",

    # Metrics: Accuracy is chosen as the primary metric to evaluate the performance of the model because it directly measures the proportion of correctly 
    # classified images out of the total number of images.
    metrics = ["accuracy"]
)

# 2) COMPILING THE INTERMIDIATE CNN MODEL --------------------------------------------------------------------------------------
print("\nCompiling the Intermediate CNN model...")

# We use the same techiques for compiling the intermediate model as the baseline model because they are both suitable for our 
# multi-class classification task and provide a good starting point for training.
model_2_intermidiate.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

# 3) COMPILING THE ADVANCED CNN MODEL --------------------------------------------------------------------------------------
print("\nCompiling the Advanced CNN model...")

# We use the same techiques for compiling the advanced model as the baseline and intermediate models because they are all 
# suitable for our multi-class classification task and provide a good starting point for training.
model_3_advanced.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

# ------------------------------------------------------------ TRAINING THE 3 MODELS ----------------------------------------------------------------------
# 1) TRAINING THE BASELINE CNN MODEL --------------------------------------------------------------------------------------
# First we define callbacks to block the training proces if we see that there is not improvement between two callbacks
callbacks_1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor = "val_accuracy",
        patience = 5,
        restore_best_weights = True
    )
]

print("\nTraining the Baseline CNN model...")

# Training configuration (batch size was not specified because already defined during the loading phase, and the number of epochs is set to 20 which is a common 
# choice for training CNNs on image classification tasks. This allows the model to learn from the data while also providing enough iterations for convergence 
# without overfitting.)
EPOCHS = 20

# Training the model
history_model_1 = model_1_baseline.fit(
    train_data,                      # training the dataset
    epochs = EPOCHS,                 # number of epochs to train
    validation_data = val_data,      # validation dataset to evaluate the model's performance after each epoch
    verbose = 1,                      # shows the training progress and metrics for each epoch (1 = progress bar, 2 = one line per epoch, 0 = silent)
    callbacks = callbacks_1
)

print ("\nBaseline CNN model training completed.")

# Summary of the training results for the baseline model
final_train_loss = history_model_1.history["loss"][-1]
final_train_accuracy = history_model_1.history["accuracy"][-1]
final_val_loss = history_model_1.history["val_loss"][-1]
final_val_accuracy = history_model_1.history["val_accuracy"][-1]

print(f"Final Training Loss: {round(final_train_loss, 2)}")
print(f"Final Training Accuracy: {round(final_train_accuracy, 2)}")
print(f"Final Validation Loss: {round(final_val_loss, 2)}")
print(f"Final Validation Accuracy: {round(final_val_accuracy, 2)}")

# Check for presence of overfitting
accuracy_gap = final_train_accuracy - final_val_accuracy
print (accuracy_gap)

if accuracy_gap > 0.1:   # More than 10% gap between training and validation accuracy is a strong indicator of overfitting
    print("Warning: Potential overfitting detected (accuracy gap > 10%). Consider implementing regularization techniques or collecting more data.")
elif accuracy_gap < 0.05:   # 5-10% gap is generally acceptable, but less than 5% is ideal
    print("Good: No significant overfitting detected (accuracy gap < 5%). The model is generalizing well to the validation data.")
else:
    print("No significant overfitting detected (gap < 5%).")

# Results:
# Although data augmentation was optional, it was implemented in the training pipeline to enhance the diversity of the training data and improve the model's generalization capabilities.
# The effects of data augmentation can be observed in the training and validation accuracy trends. If the training accuracy is significantly higher than the validation accuracy, it may 
# indicate that the model is overfitting to the augmented training data. However, if both training and validation accuracies are improving and relatively close to each other, it suggests 
# that the data augmentation is helping the model learn more robust features without overfitting. In this case, we can conclude that the data augmentation techniques implemented in the 
# training pipeline have contributed positively to the model's performance on the rock-paper-scissors classification task, as evidenced by the training and validation accuracy trends 
# observed during the training process. In fact, in our case training accuracy (85%) was lower than the validation accuracy (97%), which is expected when using augmentation techniques, 
# and it indicates that the model is generalizing well to the validation data without overfitting to the augmented training data.

# 2) TRAINING THE INTERMIDIATE CNN MODEL --------------------------------------------------------------------------------------
callbacks_2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor = "val_accuracy",
        patience = 10,
        restore_best_weights = True
    )
]

print("\nTraining the Intermediate CNN model...")

# Training configuration is the same as the baseline model to ensure a fair comparison between the two architectures.
EPOCHS = 20

# Training the model
history_model_2 = model_2_intermidiate.fit(
    train_data,                   # Training dataset
    epochs = EPOCHS,              # 20 epochs
    validation_data = val_data,   # validation dataset
    verbose = 1,                   # shows the progress
    callbacks = callbacks_2
)

print("\nIntermidiate CNN model training complete")

# Summary of results
final_train_loss_2 = history_model_2.history["loss"][-1]
final_train_accuracy_2 = history_model_2.history["accuracy"][-1]
final_val_loss_2 = history_model_2.history["val_loss"][-1]
final_val_accuracy_2 = history_model_2.history["val_accuracy"][-1]

print(f"Final Training Loss: {round(final_train_loss_2, 2)}")
print(f"Final Training Accuracy: {round(final_train_accuracy_2, 2)}")
print(f"Final Validation Loss: {round(final_val_loss_2, 2)}")
print(f"Final Validation Accuracy: {round(final_val_accuracy_2, 2)}")

# Checking for the presense of overfitting
accuracy_gap_2 = final_train_accuracy_2 - final_val_accuracy_2
print(accuracy_gap_2)

if accuracy_gap_2 > 0.1:   # More than 10% gap between training and validation accuracy is a strong indicator of overfitting
    print("Warning: Potential overfitting detected (accuracy gap > 10%). Consider implementing regularization techniques or collecting more data.")
elif accuracy_gap_2 < 0.05:   # 5-10% gap is generally acceptable, but less than 5% is ideal
    print("Good: No significant overfitting detected (accuracy gap < 5%). The model is generalizing well to the validation data.")
else:
    print("No significant overfitting detected (gap < 5%).")

# Results:
# Model 2 demonstates a greater performance across all metrics if we compare it with the resutls of model 1:
# - Higher validation accuracy 
# - Significantly higher training accuracy 
# - Lower validation loss 
# - Accuracy gap reduced (meaning better training stability)

# The addition of the third convolutional block with 128 filters and dropout regularization of 0.5, allowed the model to learn more complex features
# while mantaining great generalization. The smaller accuracy gap indicates that model 2 handles data augmentation more efficiently while achieving 
# higher performance on both training and validation sets

# 3) TRAINING THE ADVANCED CNN MODEL --------------------------------------------------------------------------------------------------------------------
callbacks_3 = [
    tf.keras.callbacks.EarlyStopping(
        monitor = "val_accuracy",
        patience = 10,
        restore_best_weights = True
    )
]

print("\nTraining the advanced CNN model...")

# Training configuration is the same as the baseline and intermidiate model to ensure a fair comparison between the two architectures.
# In this case we increased the epochs to 50 instead of 20 because BatchNormalization requires more warmup time with lower learning rate (0.0001). The 
# first 12 epochs are used for BatchNorm statistics stabilization, leaving almost 38 effective epochs for learning
EPOCHS = 50

# Training the model
history_model_3 = model_3_advanced.fit(
    train_data,                   # Training dataset
    epochs = EPOCHS,              # 50 epochs
    validation_data = val_data,   # validation dataset
    verbose = 1,                   # shows the progress
    callbacks = callbacks_3
)

print("\nAdvanced CNN model training complete")

# Summary of results
final_train_loss_3 = history_model_3.history["loss"][-1]
final_train_accuracy_3 = history_model_3.history["accuracy"][-1]
final_val_loss_3 = history_model_3.history["val_loss"][-1]
final_val_accuracy_3 = history_model_3.history["val_accuracy"][-1]

print(f"Final Training Loss: {round(final_train_loss_3, 2)}")
print(f"Final Training Accuracy: {round(final_train_accuracy_3, 2)}")
print(f"Final Validation Loss: {round(final_val_loss_3, 2)}")
print(f"Final Validation Accuracy: {round(final_val_accuracy_3, 2)}")

# Checking for presence of overfitting
accuracy_gap_3 = final_train_accuracy_3 - final_val_accuracy_3
print(accuracy_gap_3)

if accuracy_gap_3 > 0.1:   # More than 10% gap between training and validation accuracy is a strong indicator of overfitting
    print("Warning: Potential overfitting detected (accuracy gap > 10%). Consider implementing regularization techniques or collecting more data.")
elif accuracy_gap_3 < 0.05:   # 5-10% gap is generally acceptable, but less than 5% is ideal
    print("Good: No significant overfitting detected (accuracy gap < 5%). The model is generalizing well to the validation data.")
else:
    print("No significant overfitting detected (gap < 5%).")

# ----------------------------------------------------------- CROSS VALIDATION OF ALL THE MODELS --------------------------------------------------------------------------------------------
# Cross validation is performed on all 3 models to obtain a more robust and unbiaed estimate of their generalization performance. This implementation uses StratifiedKFold to ensure balanced 
# class distribution across all folds, and applies the same data augmentation pipeline used in main training for consistency.

# Key design decisions:
# - StratifiedKFold instead of KFold: Guarantees proportional class representation in each fold 
# - Data augmentation applied: Cross validation now reflects the actual training setup (augmented data)
# - Train and validation sets are merged for CV: Uses th full non-test pool (almost 85% of data) for more reliable estimates
# - Test set untouched: Never used during Cross Validation, reserved for final evaluation only

# For cross validation 10 epochs will be used instead of the 20/50 used before. Using 10 epoches keeps runtime practical. The goal here is robust comparative evaluation, not peak accuracy - 
# that was already achieved in the full training runs above. But for model_3 we will still use at least 30 epochs

# Extracting train + validation data into numpy arrays --------------------------------------------------------------------------------------------------------------------------------------
X_cross_val, Y_cross_val = [], []

# Building fresh dataset from the file paths to avoid any pipeline state issues 
train_data_cv = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(load_and_resize, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (normalization_layer(x), y))
)

val_data_cv = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(load_and_resize, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (normalization_layer(x), y))
)

for images, labels in train_data_cv:
    X_cross_val.append(images.numpy())
    Y_cross_val.append(labels.numpy())

for images, labels in val_data_cv:
    X_cross_val.append(images.numpy())
    Y_cross_val.append(labels.numpy())

X_cross_val = np.concatenate(X_cross_val, axis = 0)
Y_cross_val = np.concatenate(Y_cross_val, axis = 0)

print(f"Cross Validation pool: {X_cross_val.shape[0]} samples")
print(f"X shape: {X_cross_val.shape}, Y shape: {Y_cross_val.shape}")

# Defining model factory functions -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Each factory recreates the model from scratch for every fold to ensure complete independence. This prevents weight leakage across folds, which would produce optimitically biased results.

def build_modeL_1():
    """Baseline CNN: 2 convolutional blocks, no dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (150, 150, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dense(3, activation = "softmax"),
    ], name = "Baseline_CNN_CV")
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return model

def build_modeL_2():
    """Intermediate CNN: 3 convolutional blocks + dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (150, 150, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation = "softmax"),
    ], name = "Intermediate_CNN_CV")
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return model

def build_modeL_3():
    """Intermediate CNN: 3 convolutional blocks + dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (150, 150, 3)),
        tf.keras.layers.Conv2D(32, 3, padding = "same", use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding = "same", use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding = "same", use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, padding = "same", use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation = "softmax"),
    ], name = "Advanced_CNN_CV")
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return model

# 5-fold Cross Validation loop ---------------------------------------------------------------------------------------------------------------------------------------------------------------
K_FOLDS = 5

# Different CV epochs for each model
CV_EPOCHS_MODEL_1 = 10
CV_EPOCHS_MODEL_2 = 10
CV_EPOCHS_MODEL_3 = 30

kf = StratifiedKFold(n_splits = K_FOLDS, shuffle = True, random_state = 42)

cv_results = {
    "model_1": {"accuracy": [], "loss": []},
    "model_2": {"accuracy": [], "loss": []},
    "model_3": {"accuracy": [], "loss": []},
}

model_builders = {
    "model_1": build_modeL_1,
    "model_2": build_modeL_2,
    "model_3": build_modeL_3,
}

# Map each model to its specific CV epochs
cv_epochs_map = {
    "model_1": CV_EPOCHS_MODEL_1,
    "model_2": CV_EPOCHS_MODEL_2,
    "model_3": CV_EPOCHS_MODEL_3
}

for model_key, builder_fn in model_builders.items():
    cv_epochs = cv_epochs_map[model_key]
    print(f"Cross Validation: {model_key.upper().replace('_', '')} ({cv_epochs} epochs per fold)")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_cross_val, Y_cross_val), start = 1):
        print(f"\nFold {fold_idx}/{K_FOLDS}:")

        X_fold_train, X_fold_val = X_cross_val[train_idx], X_cross_val[val_idx]
        Y_fold_train, Y_fold_val = Y_cross_val[train_idx], Y_cross_val[val_idx]

        # Fresh model for every fold which prevents any weight leakage between folds
        fold_model = builder_fn()

        # Creating tf.data datasets from numpy arrays to apply augmentation
        # This makes CV consistent with the actual training pipeline
        fold_train_ds = (
            tf.data.Dataset.from_tensor_slices((X_fold_train, Y_fold_train))
            .batch(BATCH_SIZE)
            .map(lambda x, y: (data_augmentation(x, training = True), y), num_parallel_calls = tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        fold_val_ds = (
            tf.data.Dataset.from_tensor_slices ((X_fold_val, Y_fold_val))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        fold_model.fit(
            fold_train_ds,
            epochs = cv_epochs,
            validation_data = fold_val_ds,
            verbose = 0
        )

        fold_loss, fold_accuracy = fold_model.evaluate(fold_val_ds, verbose = 0)

        cv_results[model_key]["accuracy"].append(fold_accuracy)
        cv_results[model_key]["loss"].append(fold_loss)

        print(f"Fold {fold_idx} Accuracy: {fold_accuracy} | Loss: {fold_loss}")

        del fold_model
    # Free GPU/CPU memory after each fold to avoidc out of memory errors across 15 training runs    
    tf.keras.backend.clear_session()

# Cross validation results summary ----------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nCROSS VALIDATION RESULTS SUMMARY:\n")
cv_summary = {}
for model_key in ["model_1", "model_2", "model_3"]:
    accs = cv_results[model_key]["accuracy"]
    losses = cv_results[model_key]["loss"]
    mean_acc =np.mean(accs)
    std_acc = np.std(accs)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)

    cv_summary[model_key] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "all_fold_accuracies": accs
    }

    print(f"\n{model_key.upper().replace('_', '')}:")
    print(f"Mean CV Accuracy: {mean_acc:.4f}")
    print(f"Mean CV loss: {mean_loss:.4f}")
    print(f"Per-Fold Scores: {[round(a, 4) for a in accs]}")
    print(f"Stability: {'Stable' if std_acc < 0.03 else 'Some variance across folds'}")

# Visualization of cross-validation results -------------------------------------------------------------------------------------------------------------------------------------------------
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize = (14, 6))

model_labels = ["Model 1\n(Baseline)", "Model_2\n(Intermidiate)", "Model_3\n(Advanced)"]
colors = ["yellow", "green", "blue"]
fold_x = np.arange(1, K_FOLDS + 1)

# Plot 1: Per-fold accuracy line chart
for model_key, label, color in zip (["model_1", "model_2", "model_3"], model_labels, colors):
    axes[0].plot(fold_x, cv_results[model_key]["accuracy"],
                 marker = "o", label = label, color = color, linewidth = 2, markersize = 6)

axes[0].set_title ("Per-Fold Validation Accuracy", fontsize = 13, fontweight = "bold")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Accuracy")
axes[0].set_xticks(fold_x)
axes[0].legend()
axes[0].grid(True, alpha = 0.3)
axes[0].set_ylim([0, 1.05])

# Plot 2: Mean accuracy bar chart with standard deviation error bars
mean_accs = [cv_summary[k]["mean_accuracy"] for k in ["model_1", "model_2", "model_3"]]
std_accs = [cv_summary[k]["std_accuracy"] for k in ["model_1", "model_2", "model_3"]]

bars = axes[1].bar(model_labels, mean_accs, color = colors, alpha = 0.8,
                   yerr = std_accs, capsize = 6, edgecolor = "black", linewidth = 1.2)

for bar, mean, std in zip (bars, mean_accs, std_accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std + 0.01,
                 f"{mean.round(2)} +- {std.round(2)}",
                 ha = "center", va = "bottom", fontsize = 9, fontweight = "bold")
    
axes[1].set_title("Mean CV Accuracy +- Std Deviation", fontsize = 9, fontweight = "bold")
axes[1].set_ylabel("Mean Accuracy")
axes[1].set_ylim([0, 1.15])
axes[1].grid(True, alpha = 0.3, axis = "y")

plt.suptitle("5-Fold Cross-Validation Comparison: All 3 CNN models",
             fontsize = 14, fontweight = "bold")
plt.tight_layout()
plt.show(block = False); plt.pause(3)

print("\nCross Validation complete!")

# -------------------------------------------------------- HYPERPARAMETER TUNING FOR MODEL 2 ------------------------------------------------------------------------------
# Hyperparameter tuning is performed on Model 2 (Intermidiate CNN) to find the optimal combination of hyperparameters that maximize validation accuracy. We'll tune the
# following hyperparameters:
# - 1) Learning rate: Controls the step size during gradient descent
# - 2) Dropout rate: Controls regularization strength to prevent overfitting
# - 3) Number of filters in Conv layers: Controls model capacity
# - 4) Dense layer size: Controls the complexity of the classification head

# Strategy: Grid search with 3-fold cross-validation on a subset of hyperparameter combinations. We use a smaller k = 3 for Cross Validation here (instead of 5) to keep 
# runtime practical during grid search.

print("\nHYPERPARAMETER TUNING MODEL 2 INTERMIDIATE")

# Defining the hyperparameter grid to search
# We keep the grid manageable (2-3 values per parameter) to avoid combinatorial explosion
hyperparameter_grid = {
    "learning_rate": [0.0001, 0.001, 0.01],           # low, medium, high
    "dropout_rate": [0.3, 0.5, 0.7],                  # low, medium, high
    "conv_filters": [(32, 64, 128), (64, 128, 256)],  # original vs double capacity
    "dense_units": [64, 128, 256]
}

# Calculation of the total combinations
total_combinations = (
    len(hyperparameter_grid["learning_rate"]) *
    len(hyperparameter_grid["dropout_rate"]) *
    len(hyperparameter_grid["conv_filters"]) *
    len(hyperparameter_grid["dense_units"])
)

print (f"Hyperparameter search space:")
print(f" - Learning rates: {hyperparameter_grid['learning_rate']}")
print(f" - Dropout rates: {hyperparameter_grid['dropout_rate']}")
print(f" - Conv filter configurations: {hyperparameter_grid['conv_filters']}")
print(f" - Dense layer sizes: {hyperparameter_grid['dense_units']}")
print (f"\nTotal combinations to test: {total_combinations}")
print(f"Each combination uses 3-fold CV with 10 epochs per fold")
print(f"Estimated runtime: almost {total_combinations * 3 * 10 * 4 // 60} minutes\n")

# Model builder function with configurable hyperprameters
def build_tuned_model_2 (learning_rate, dropout_rate, conv_filters, dense_units):
    """Build Model 2 with specified hyperparameters.
    
    Args:
        learning_rate: Optimizer learning rate
        dropout_rate: Dropout probability (0-1)
        conv_filters: Tuple of (filters_layer1, filters_layer2, filters_layer3)
        dense_units: Number of units in the dense layer
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (150, 150, 3)),

        # Convolutional block 1
        tf.keras.layers.Conv2D(conv_filters[0], (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Convolutional block 2
        tf.keras.layers.Conv2D(conv_filters[1], (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Convolutional block 3
        tf.keras.layers.Conv2D(conv_filters[2], (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classification head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation = 'relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(3, activation = 'softmax')
    ], name = 'Tuned_intermidiate_CNN')

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model

# Grid search with 3-fold cross validation
K_FOLDS_TUNING = 3
TUNING_EPOCHS = 10

kf_tuning = StratifiedKFold(n_splits = K_FOLDS_TUNING, shuffle = True, random_state = 42)

# Storing the results for all hyperparameter combinations
tuning_results = []

# Generate all combinations
all_combinations = list(itertools.product(
    hyperparameter_grid['learning_rate'],
    hyperparameter_grid['dropout_rate'],
    hyperparameter_grid['conv_filters'],
    hyperparameter_grid['dense_units']
))

for idx, (lr, dropout, filters, dense) in enumerate(all_combinations, start = 1):
    print(f"[{idx}/{total_combinations}] Testing: LR = {lr}, Dropout = {dropout}, Filters = {filters}, Dense = {dense}")

    fold_accuracies = []
    fold_losses = []

    # 3-fold cross validation for this hyperparameters combination
    for fold_idx, (train_idx, val_idx) in enumerate(kf_tuning.split(X_cross_val, Y_cross_val), start = 1):
        X_fold_train, X_fold_val = X_cross_val[train_idx], X_cross_val[val_idx]
        Y_fold_train, Y_fold_val = Y_cross_val[train_idx], Y_cross_val[val_idx]

        # Building the model with the current hyperparameters
        tuned_model = build_tuned_model_2(lr, dropout, filters, dense)

        # Create datasets with augmentation
        fold_train_ds = (
            tf.data.Dataset.from_tensor_slices((X_fold_train, Y_fold_train))
            .batch(BATCH_SIZE)
            .map(lambda x, y: (data_augmentation(x, training = True), y), num_parallel_calls = tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        fold_val_ds = (
            tf.data.Dataset.from_tensor_slices((X_fold_val, Y_fold_val))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Train
        tuned_model.fit(
            fold_train_ds,
            epochs = TUNING_EPOCHS,
            validation_data = fold_val_ds,
            verbose = 0
        )

        # Evaluate
        fold_loss, fold_accuracy = tuned_model.evaluate(fold_val_ds, verbose = 0)
        fold_accuracies.append(fold_accuracy)
        fold_losses.append(fold_loss)

        del tuned_model

    # Calculate mean performance across folds
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)

    # Storing the results 
    tuning_results.append({
        'learning_rate': lr,
        'dropout_rate': dropout,
        'conv_filters': filters,
        'dense_units': dense,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_loss': mean_loss,
        'fold_accuracies': fold_accuracies
    })

    print(f"Mean CV Accuracy: {mean_accuracy:.4f} +- {std_accuracy:.4f}")

    # Clear session after each hyperparameter combination
    tf.keras.backend.clear_session()

# Analysis of the results
print("\nHYPERPARAMETER TUNING RESULTS:")

# We sort the results by mean accuracy (descending order)
tuning_results_sorted = sorted(tuning_results, key = lambda x: x['mean_accuracy'], reverse = True)

# Top 5 configurations
print("\nTop 5 Hyperparameter Configurations:")
for rank, result in enumerate(tuning_results_sorted[:5], start = 1):
    print(f"{rank}. Mean Accuracy: {result['mean_accuracy']:.4f} +- {result['std_accuracy']:.4f}")
    print(f" - Learning_rate: {result['learning_rate']}")
    print(f" - Dropout_rate: {result['dropout_rate']}")
    print(f" - Conv Filters: {result['conv_filters']}")
    print(f" - Dense Units: {result['dense_units']}")
    print(f" - Mean Loss: {result['mean_loss']:.4f}")
    print(f" - Per-Fold Accuracies: {[round(a, 4) for a in result['fold_accuracies']]}\n")

# Best configuration
best_config = tuning_results_sorted[0]
print("\nBEST HYPERPARAMETER CONFIGURATION:")
print(f"Learning_rate: {best_config['learning_rate']}")
print(f" - Dropout_rate: {best_config['dropout_rate']}")
print(f" - Conv Filters: {best_config['conv_filters']}")
print(f" - Dense Units: {best_config['dense_units']}")
print(f" - Mean CV accuracy: {best_config['mean_accuracy']:.4f} +- {best_config['std_accuracy']:.4f}")
print(f" - Mean CV Loss: {best_config['mean_loss']:.4f}")
print(f" - Per-Fold Accuracies: {[round(a, 4) for a in best_config['fold_accuracies']]}\n")

# Comparison withe the original Model 2 configuration
original_config_accuracy = 0.9242    # From earlier CV results
improvement = (best_config['mean_accuracy'] - original_config_accuracy) * 100

print("COMPARISON WITH ORIGINAL MODEL 2:")
print(f"Original Model 2 Accuracy: {original_config_accuracy:.4f}")
print(f"Tuned Model 2 CV Accuracy: {best_config['mean_accuracy']:.4f}")
print(f"Improvement: {improvement:+.2f} %")

if improvement > 0:
    print(f"\nHypermarameter tuning improved Model 2 performance by {improvement:.2f} %")
else:
    print(f"\nOriginal Hyperparameters were already near optimal (difference: {improvement:.2f} %)")

# ------------------------------------------------------------- COMPARING ALL THE 3 MODELS -----------------------------------------------------------------------------------------
print("\nCOMPATING ALL 3 MODELS ON VALIDATION SET")

# Evaluate all 3 original models on validation set
models_comparison = {
    "Model_1 (Baseline)": model_1_baseline,
    "Model_2 (Intermidiate)": model_2_intermidiate,
    "Model_3 (Advanced)": model_3_advanced
}

for model_name, model in models_comparison.items():
    print(f"\n {model_name}")

    # Get predictions on validation set
    val_predictions_probs = model.predict(val_data, verbose = 0)
    val_predictions = np.argmax(val_predictions_probs, axis = 1)

    # Calculation of the metrics
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_precision = precision_score(val_labels, val_predictions, average = "weighted")
    val_recall = recall_score(val_labels, val_predictions, average = "weighted")
    val_f1 = f1_score(val_labels, val_predictions, average = "weighted")

    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1-score: {val_f1:.4f}")

# ----------------------------------------------------------------- FINAL TEST SET EVALUATION -----------------------------------------------------------------------------------------------------
# Since both model 2 tuned and model 3 (wthout tuning) perform really well, e will do the test set evaluation on both of them to let the results decide which one its the best

# 1) MODEL 3 (BEST VALIDATION) --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Now first, we evaluate the model 3 on the test set
test_pred_model_3 = np.argmax(model_3_advanced.predict(test_data, verbose = 0), axis = 1)

accuracy_3 = accuracy_score(test_labels, test_pred_model_3)
precision_3 = precision_score(test_labels, test_pred_model_3, average = 'weighted')
recall_3 = recall_score(test_labels, test_pred_model_3, average = 'weighted')
f1_score_3 = f1_score(test_labels, test_pred_model_3, average = 'weighted')

print("\nTest set metrics:")
print(f"Accuracy: {accuracy_3:.4f}")
print(f"Precision: {precision_3:.4f}")
print(f"Recall: {recall_3:.4f}")
print(f"F1-score: {f1_score_3:.4f}")

# 2) MODEL 2 (WITHOUT TUNING) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Use already trained model 2
test_pred_model_2 = np.argmax(model_2_intermidiate.predict(test_data, verbose = 0), axis = 1)

accuracy_2 = accuracy_score(test_labels, test_pred_model_2)
precision_2 = precision_score(test_labels, test_pred_model_2, average = 'weighted')
recall_2 = recall_score(test_labels, test_pred_model_2, average = 'weighted')
f1_score_2 = f1_score(test_labels, test_pred_model_2, average = 'weighted')

print("\nTest set metrics:")
print(f"Accuracy: {accuracy_2:.4f}")
print(f"Precision: {precision_2:.4f}")
print(f"Recall: {recall_2:.4f}")
print(f"F1-score: {f1_score_2:.4f}")

# So it has a worse performance on the test set compared to the third model 

# 3) MODEL 2 (TUNED) ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Model 2 (intermidiate) - Hyperparameter Tuned & Retrained on Train + Val")
print("\nBest hyperparameters from grid search:")
print(f" - Learning rate: {best_config['learning_rate']}")
print(f" - Dropout rate: {best_config['dropout_rate']}")
print(f" - Conv filters: {best_config['conv_filters']}")
print(f" - Dense Units: {best_config['dense_units']}")

# Building and training the model on combined data
final_model = build_tuned_model_2(
    learning_rate = best_config['learning_rate'],
    dropout_rate = best_config['dropout_rate'],
    conv_filters = best_config['conv_filters'],
    dense_units = best_config['dense_units']
)

print("\nTraining on combined train + validation data")
combined_train_data = train_data.concatenate(val_data)

final_history = final_model.fit(
    combined_train_data,
    epochs = 20,
    verbose = 1,
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 5, restore_best_weights = True)] 
)

test_pred_model_2_tuned = np.argmax(final_model.predict(test_data, verbose = 0), axis = 1)

accuracy_2_tuned = accuracy_score(test_labels, test_pred_model_2_tuned)
precision_2_tuned = precision_score(test_labels, test_pred_model_2_tuned, average = 'weighted')
recall_2_tuned = recall_score(test_labels, test_pred_model_2_tuned, average = 'weighted')
f1_score_2_tuned = f1_score(test_labels, test_pred_model_2_tuned, average = 'weighted')

print(f"\nAccuracy: {accuracy_2_tuned:.4f}")
print(f"Precision: {precision_2_tuned:.4f}")
print(f"Recall: {recall_2_tuned:.4f}")
print(f"F1-score: {f1_score_2_tuned:.4f}")

# ------------------------------------------------------------ COMPARISON OF RESULTS ---------------------------------------------------------------------------------------------------------------------
# Now we see the results and make a comparison of the performances on the test sets

print("\nTEST SET COMPARISON:")
print(f"Model 3 (Advanced, untuned): {accuracy_3:.4f}")
print(f"Model 2 (Intermidiate, original): {accuracy_2:.4f}")
print(f"Model 2 (Intermidiate, Tuned + retrained): {accuracy_2_tuned:.4f}")

# So, we demonstrated that a well tuned simpler model can outperform a complex model with default hyperparameters and in our case we can see that the tuning worked really well because it improved the 
# the accuracy making model tuned 2 also better then model 3 untuned

# ------------------------------------------------------------ TRAINING CURVES VISUALIZATION -------------------------------------------------------------------------------------------------------------
