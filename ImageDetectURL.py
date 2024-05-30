import os
import re
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import logging

# Load the pretrained VGG16 model
model = VGG16(weights='imagenet')

# Define the list of cat breeds to check
cat_breeds = ['tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat']

def download_image(url, folder_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Extract the filename and sanitize it
        filename = os.path.basename(url)
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        file_path = os.path.join(folder_path, sanitized_filename)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return file_path
    return None

def scrape_images_from_website(url, folder_path):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    image_tags = soup.find_all('img')
    image_urls = [urljoin(url, img['src']) for img in image_tags if 'src' in img.attrs]
    
    # Filter out non-http/https URLs
    image_urls = [img_url for img_url in image_urls if img_url.startswith(('http://', 'https://'))]

    downloaded_images = []
    for img_url in image_urls:
        downloaded_image = download_image(img_url, folder_path)
        if downloaded_image:
            downloaded_images.append(downloaded_image)
    return downloaded_images

def is_cat_breed(img_path, cat_breeds):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    logging.info(f"Predictions for {img_path}: {decoded_predictions}")
    for _, label, _ in decoded_predictions:
        if label in cat_breeds:
            return True
    return False

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

def find_matches(scraped_images, reference_images):
    matches = []
    for scraped_image in scraped_images:
        for ref_image in reference_images:
            if compare_images(scraped_image, ref_image) > 10:  # Adjust the value as needed
                matches.append(scraped_image)
                break
    return matches

logging.basicConfig(filename='output.log', level=logging.INFO)

def main(reference_images, website_url, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    logging.info(f"Scraping images from website: {website_url}")
    scraped_images = scrape_images_from_website(website_url, folder_path)
    logging.info(f"Downloaded images: {scraped_images}")
    
    cat_breed_images = [img for img in scraped_images if is_cat_breed(img, cat_breeds)]
    logging.info(f"Cat breed images found: {cat_breed_images}")
    
    matches = find_matches(cat_breed_images, reference_images)
    logging.info("Images matching reference patterns: %s", matches)
    return matches


# Usage
reference_images = ['\\images\\cat1.jpg', '\\images\\cat2.jpg']  # Add reference image paths
website_url = 'https://portalvet.royalcanin.com.br/guia-de-racas/gato-persa'  # Add the website URL to scrape
folder_path = '\\downloaded_image'  # Folder to save downloaded images
matching_images = main(reference_images, website_url, folder_path)
print("Images matching reference patterns:", matching_images)
