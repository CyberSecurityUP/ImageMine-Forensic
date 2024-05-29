# Nude Detector Model

This project is a deep learning-based image classifier that identifies nudity in images. It utilizes the VGG16 convolutional neural network architecture, pretrained on the ImageNet dataset, and fine-tuned for binary classification.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Practical Usage](#practical-usage)
- [License](#license)

## Installation

To run this project, ensure you have Python installed along with the following libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- OpenCV

Install the required packages using pip:
```bash
pip install tensorflow keras numpy matplotlib pillow opencv-python
```

## Dataset

The dataset should be organized into three directories: `train`, `validation`, and `test`, each containing subdirectories for the binary classes (`class1`, `class2`). Replace `class1` and `class2` with appropriate names such as `nude` and `not_nude`.

Example structure:
```
E:\
|-- train
|   |-- class1
|   |-- class2
|-- validation
|   |-- class1
|   |-- class2
|-- test
|   |-- class1
|   |-- class2
```

## Usage

The model is defined and trained using the provided code. Save this script as `train_nude_detector.py` and run it in your Python environment.

```python
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
train_dir = 'E:\\train'
validation_dir = 'E:\\validation'
test_dir = 'E:\\test'

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Load pretrained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy}')

# Save the trained model
model.save('nudes_detector_model.h5')
```

## Model Architecture

The model uses the VGG16 architecture without the top classification layer. Custom layers are added:
- Global Average Pooling layer
- Dense layer with 1024 units and ReLU activation
- Output Dense layer with sigmoid activation for binary classification

## Training

The model is trained for 10 epochs using the Adam optimizer and binary crossentropy loss. Data augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping are applied to the training data.

## Evaluation

The model is evaluated on a separate test dataset, and the accuracy is printed to the console.

## Saving the Model

The trained model is saved as `nudes_detector_model.h5` for later use or deployment.

## Practical Usage

You can use the trained model to detect nudity in images and compare images for potential matches. Save this script as `nude_detector_usage.py` and run it in your Python environment.

```python
import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import logging

# Load the pretrained and fine-tuned VGG16 model
model = load_model('pornography_detector_model.h5')

def is_pornographic(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction = model.predict(x)
    logging.info(f"Prediction for {img_path}: {prediction}")
    return prediction[0][0] > 0.5  # Adjust threshold as needed

def check_folder_for_pornography(folder_path):
    porn_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            if is_pornographic(file_path):
                porn_images.append(file_path)
    return porn_images

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    logging.info(f"Comparison between {img1_path} and {img2_path}: {len(matches)} matches")
    return len(matches)

def find_matches(porn_images, reference_images):
    matches = []
    for porn_image in porn_images:
        for ref_image in reference_images:
            if compare_images(porn_image, ref_image) > 10:  # adjust the value as needed
                matches.append(porn_image)
                break
    return matches

logging.basicConfig(filename='output.log', level=logging.INFO)

def main(reference_images, folder_path):
    logging.info(f"Checking reference images: {reference_images}")
    for ref_image in reference_images:
        if not os.path.exists(ref_image):
            logging.error(f"Reference image not found: {ref_image}")
        else:
            logging.info(f"Reference image found: {ref_image}")

    porn_images = check_folder_for_pornography(folder_path)
    logging.info(f"Pornographic images found: {porn_images}")
    matches = find_matches(porn_images, reference_images)
    logging.info("Images matching reference patterns: %s", matches)
    return matches

# Usage
reference_images = ['path_to_reference_image1.jpg', 'path_to_reference_image2.jpg']  # Add reference image paths
folder_path = 'path_to_folder_to_check'  # Add the folder path to check
matching_images = main(reference_images, folder_path)
print("Images matching reference patterns:", matching_images)
```

## License

This project is licensed under the MIT License.

---

Feel free to modify the code and adapt it to your specific needs. Contributions are welcome!

For any issues or questions, please open an issue on the [GitHub repository](https://github.com/yourusername/nude-detector).
