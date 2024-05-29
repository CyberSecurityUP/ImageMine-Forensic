import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import logging

# Load the pretrained and fine-tuned VGG16 model
model = load_model('nudes_detector_model.h5')

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
