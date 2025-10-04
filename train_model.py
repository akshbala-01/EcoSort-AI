import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import numpy as np

# --- 1. Define Corrected Data Paths ---
# CRITICAL: Replace 'C:\Users\HP\Desktop\EcoSort-AI' with the EXACT absolute path to your project folder.
# Ensure there is NO backslash at the end.
PROJECT_ROOT = r'C:\Users\HP\Desktop\EcoSort-AI' 

# The base data directory is the folder that contains the class subfolders (cardboard, glass, etc.)
# Based on visual evidence, this is the 'dataset-resized' folder.
BASE_DATA_DIR = os.path.join(
    PROJECT_ROOT, 
    'dataset', 
    'archive', 
    'dataset-resized' 
)

# --- Path Check for Debugging ---
print(f"--- PATH CHECK ---")
print(f"Training Data Path: {BASE_DATA_DIR}")
print(f"------------------\n")


# --- 2. Configuration ---
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20 # Can be adjusted
NUM_CLASSES = 6 # cardboard, glass, metal, paper, plastic, trash

# --- 3. Data Augmentation and Preprocessing ---
# We use one generator setup with a validation_split to divide the data internally,
# since your directory structure does not have separate train/validation folders.
datagen = ImageDataGenerator(
    rescale=1./255, # Normalize
    validation_split=0.2, # Reserve 20% of data for validation
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- 4. Load Data Generators ---
try:
    # Generator for Training Data (80% of images)
    train_generator = datagen.flow_from_directory(
        BASE_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Generator for Validation Data (20% of images)
    validation_generator = datagen.flow_from_directory(
        BASE_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
except Exception as e:
    print("\n\n!!! CRITICAL ERROR: DATA FOLDER NOT FOUND !!!")
    print(f"Keras could not find the folder containing class directories at: {BASE_DATA_DIR}")
    print(f"Error Details: {e}")
    exit()

# --- 5. Build the Model (CNN Architecture) ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.5), 
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# --- 6. Compile the Model ---
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

model.summary()
print("\nStarting model training...")

# --- 7. Train the Model ---
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 8. Save the Trained Model ---
MODEL_PATH = 'models/waste_model.h5'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # Ensure 'models' folder exists
model.save(MODEL_PATH)
print(f"\nModel saved successfully at: {MODEL_PATH}")

# Note: Test evaluation is omitted since a separate test folder was not explicitly confirmed 
# inside the 'dataset-resized' directory structure.