import os
import cv2 as cv

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class DataLoader:
    def __init__(self):
        self.images = []

    def data_open(self):
        current_directory = 'dataset' 
        files = os.listdir(current_directory)
        for file in files:
            if file == '.DS_Store':  
                continue
            file_path = os.path.join(current_directory, file)
            if os.path.isdir(file_path):
                image_paths = os.listdir(file_path)
                for image_name in image_paths:
                    image_path = os.path.join(file_path, image_name)
                    if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.images.append(image_path)
        
        return self.images
    
@dataclass
class DataCleaner:
    def __init__(self,images):
        self.images = images
        self.preprocessed_images = []

    def process_single_image(self,image_path):
        img = cv.imread(image_path)
        if img is None:
            return None
        target_size = (224, 224)
        resized_img = cv.resize(img, target_size)
        gray_image = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
        _, thresholded_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY)
        return thresholded_image

    def preprocess_images_parallel(self):
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.process_single_image, self.images)
            for result in results:
                if result is not None:
                    self.preprocessed_images.append(result)
        
        return self.preprocessed_images



