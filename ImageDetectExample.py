import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import logging

model = VGG16(weights='imagenet')

cat_breeds = ['tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat']

def is_cat(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    label = decode_predictions(preds, top=1)[0][0]
    logging.info(f"Predição para {img_path}: {label}")
    return label[1] in cat_breeds

def check_folder_for_cats(folder_path):
    cat_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            if is_cat(file_path):
                cat_images.append(file_path)
    return cat_images

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    logging.info(f"Comparação entre {img1_path} e {img2_path}: {len(matches)} correspondências")
    return len(matches)

def find_matches(cat_images, reference_images):
    matches = []
    for cat_image in cat_images:
        for ref_image in reference_images:
            if compare_images(cat_image, ref_image) > 10:  # ajuste o valor conforme necessário
                matches.append(cat_image)
                break
    return matches

logging.basicConfig(filename='output.log', level=logging.INFO)

def main(reference_images, folder_path):
    logging.info(f"Verificando imagens de referência: {reference_images}")
    for ref_image in reference_images:
        if not os.path.exists(ref_image):
            logging.error(f"Imagem de referência não encontrada: {ref_image}")
        else:
            logging.info(f"Imagem de referência encontrada: {ref_image}")

    cat_images = check_folder_for_cats(folder_path)
    logging.info(f"Imagens de gatos encontradas: {cat_images}")
    matches = find_matches(cat_images, reference_images)
    logging.info("Imagens que correspondem aos gatos de referência: %s", matches)
    return matches

# Uso
reference_images = [
    'C:\\cat1.jpg',
    'C:\\cat2.jpg'
]
folder_path = 'C:\\Pictures'
matching_images = main(reference_images, folder_path)
print("Imagens que correspondem aos gatos de referência:", matching_images)
