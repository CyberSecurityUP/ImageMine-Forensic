# ImageMine-Forensic

This project is a deep learning-based image classifier that identifies nudity in images. It utilizes the VGG16 convolutional neural network architecture, pretrained on the ImageNet dataset, and fine-tuned for binary classification.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [License](#license)

## Installation

To run this project, ensure you have Python installed along with the following libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL (Pillow)

Install the required packages using pip:
```bash
pip install tensorflow keras numpy matplotlib pillow
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

## License

This project is licensed under the MIT License.

---

Feel free to modify the code and adapt it to your specific needs. Contributions are welcome!

For any issues or questions, please open an issue on the [GitHub repository](https://github.com/yourusername/nude-detector).
