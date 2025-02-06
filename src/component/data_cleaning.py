import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.component.data_loader import DataLoader

class DataCleaner:
    @staticmethod
    def process_single_image(image_path):
        """Preprocess a single image."""
        img = cv.imread(image_path)
        if img is None:
            return None
        
        # Resize
        target_size = (224, 224)
        resized_img = cv.resize(img, target_size)
        
        # Convert to Grayscale
        gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
        
        # Apply Thresholding
        #_, thresholded_img = cv.threshold(gray_img, 128, 255, cv.THRESH_BINARY)

        # Apply Gaussian Blur to reduce noise
        blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
        
        # Histogram Equalization
        equalized_img = cv.equalizeHist(blurred_img)
        
        # Normalize (Scale pixel values to [0,1])
        normalized_img = equalized_img.astype('float32') / 255.0
        
        return normalized_img

    @classmethod
    def preprocess_images_parallel(cls):
        """Preprocess images in parallel."""
        loader = DataLoader()
        images = loader.load_images()
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(cls.process_single_image, images)
        
        return [img for img in results if img is not None]



