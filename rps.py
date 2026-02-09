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


# --------------------------------------------------------------------- LOADING THE DATA ----------------------------------------------------------------------------------
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
# the training process and that no information from the test set influences the model training.

# calculate the splitting sizes of the train and test sets
total_samples = sum (counts)
train_size = int(0.70 * total_samples)
val_size = int(0.15 * total_samples)
test_size = int(0.15 * total_samples)

print(f"\nTotal samples: {total_samples}")
print(f"Train size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")

# Now we can split the data into train, validation and test sets:

# 1) Load the training data (70%) ----------------------------------------------------------------------------------------------------------------------------------------------
train_data = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels = "inferred",
    label_mode = "int",
    class_names = classes,
    batch_size = 30,
    image_size = (150, 150),
    shuffle = True,
    seed = 42,
    validation_split = 0.30,
    subset = "training"
)

# 2) Load the validation + test data (30%) --------------------------------------------------------------------------------------------------------------------------------
val_test__data = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels = "inferred",
    label_mode = "int",
    class_names = classes,
    batch_size = 30,
    image_size = (150, 150),
    shuffle = False,
    seed = 42,
    validation_split = 0.30,
    subset = "validation"
)

# 3) Now split the val_test_data into validation and test sets (15% each) -------------------------------------------------------------------------------------------------
val_batches = tf.data.experimental.cardinality(val_test__data)
val_data = val_test__data.take(val_batches // 2)
test_data = val_test__data.skip(val_batches // 2)

print (f"\nSplitting summary:")
print(f"Training batches: {tf.data.experimental.cardinality(train_data).numpy()}")
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
# Adding shuffle buffer and prefetching for improved training performance
AUTOTUNE = tf.data.AUTOTUNE

# Adding shuffle buffer to training set for better batch mixing
train_data = train_data.shuffle(buffer_size = 1000, seed = 42, reshuffle_each_iteration = True)

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
# This model employes a VGG inspird architecture with double convolutional layers in each block. The double convolutions allow the network to learn more complex feature 
# representations before spatial downsempling. We use 4 convolutional blocks with progressive filter increase (32, 64, 128, 256) to capture increasingly abstract features.
# Dropout is applied after each pooling layer to reduce overfitting, and the classification head includes a dense layer with 256 units to learn complex patterns before the final output layer.

model_3_advanced = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape = (150, 150, 3), name = "input"),

    # ---------------------- Convolutional Block 1 (VGG-inspired): --------------------------------
    # - first convolutional layer with 32 filters 
    # - 3x3 kernel size 
    # - ReLU activation to learn basic features such as edges and colors
    tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", name = "conv1a"),

    # Second convolutional layer: refine basic features before downsampling
    # Double convolution allows to learn more complex feature representations before spatial downsampling
    tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", name = "conv1b"),

    # Maxpooling layer to reduce spatial dimensions and retain important features
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool1"),
    # Dropout layer to mitigate overfitting
    tf.keras.layers.Dropout(0.5, name = "dropout1"),

    # ---------------------- Convolutional Block 2 (VGG-inspired): --------------------------------
    # 64 filters to learn more complex features such as combinations of edges and textures like hand shapes and so on ..
    # The filters where doubled compared to the first block to allow the model to capture a wider
    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu", name = "conv2a"),
    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu", name = "conv2b"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool2"),
    tf.keras.layers.Dropout(0.5, name = "dropout2"),

    # ---------------------- Convolutional Block 3 (VGG-inspired): --------------------------------
    # 128 filters to learn even more complex features and higher-level representations of the images, such as specific hand gestures and finer details
    tf.keras.layers.Conv2D(128, (3, 3), activation = "relu", name = "conv3a"),
    tf.keras.layers.Conv2D(128, (3, 3), activation = "relu", name = "conv3b"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool3"),
    tf.keras.layers.Dropout(0.5, name = "dropout3"),

    # ---------------------- Convolutional Block 4 (VGG-inspired): --------------------------------
    # 256 filters to capture highly abstract features and complex patterns in the images, such as specific hand gestures and finer details
    # Single convolutional layer in this block to reduce computational complexity while still learning high-level features, and also because
    # the spatial dimensions have been significantly reduced by the previous pooling layers, so a single convolution can effectively capture 
    # the remaining features without overfitting
    tf.keras.layers.Conv2D(256, (3, 3), activation = "relu", name = "conv4a"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "maxpool4"),
    tf.keras.layers.Dropout(0.5, name = "dropout4"),

    # Classification head:
    # Flatten layer to convert 2D feature maps to 1D feature vectors
    tf.keras.layers.Flatten(name = "flatten"),

    # First dense layer with 256 units and ReLU activation to learn complex patterns and relationships between the features extracted by the convolutional blocks
    tf.keras.layers.Dense(256, activation = "relu", name = "dense1"),
    tf.keras.layers.Dropout(0.5, name = "dropout5"),

    # Second dense layer with 128 units and ReLU activation to further learn complex patterns before the final output layer
    tf.keras.layers.Dense(128, activation = "relu", name = "dense2"),
    tf.keras.layers.Dropout(0.5, name = "dropout6"),

    # Output layer with 3 units (one for each class) and softmax activation for multi-class classification
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
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

# ------------------------------------------------------------ TRAINING THE 3 MODELS ----------------------------------------------------------------------

# 1) TRAINING THE BASELINE CNN MODEL --------------------------------------------------------------------------------------
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
    verbose = 1                      # shows the training progress and metrics for each epoch (1 = progress bar, 2 = one line per epoch, 0 = silent)
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
if accuracy_gap > 0.1:   # More than 10% gap between training and validation accuracy is a strong indicator of overfitting
    print("Warning: Potential overfitting detected (accuracy gap > 10%). Consider implementing regularization techniques or collecting more data.")
elif accuracy_gap < 0.05:   # 5-10% gap is generally acceptable, but less than 5% is ideal
    print("Good: No significant overfitting detected (accuracy gap < 5%). The model is generalizing well to the validation data.")
else:
    print("No significant overfitting detected (gap < 5%).")

# Although data augmentation was optional, it was implemented in the training pipeline to enhance the diversity of the training data and improve the model's generalization capabilities.
# The effects of data augmentation can be observed in the training and validation accuracy trends. If the training accuracy is significantly higher than the validation accuracy, it may 
# indicate that the model is overfitting to the augmented training data. However, if both training and validation accuracies are improving and relatively close to each other, it suggests 
# that the data augmentation is helping the model learn more robust features without overfitting. In this case, we can conclude that the data augmentation techniques implemented in the 
# training pipeline have contributed positively to the model's performance on the rock-paper-scissors classification task, as evidenced by the training and validation accuracy trends 
# observed during the training process. In fact, in our case training accuracy (85%) was lower than the validation accuracy (97%), which is expected when using augmentation techniques, 
# and it indicates that the model is generalizing well to the validation data without overfitting to the augmented training data.




