import os
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor

class DataLoader:
    def __init__(self):
        self.images = []

    def data_open(self):
        current_directory = 'dataset'  # Dataset directory
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
    
class DataCleaner:
    preprocessed_images = []  # Class variable to hold preprocessed images

    @classmethod
    def data_open(cls):
        """Open and load images from dataset"""
        data_loader = DataLoader()  
        return data_loader.data_open()  

    @classmethod
    def process_single_image(cls, image_path):
        """Process a single image (resize, grayscale, threshold)"""
        img = cv.imread(image_path)
        if img is None:
            return None
        target_size = (224, 224)
        resized_img = cv.resize(img, target_size)
        gray_image = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
        _, thresholded_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY)
        edges = cv.Canny(thresholded_image, 100, 200)
        
        equalized_image = cv.equalizeHist(thresholded_image)
        
      
        normalized_image = thresholded_image.astype('float32') / 255.0
        return thresholded_image

    @classmethod
    def preprocess_images_parallel(cls):
        """Preprocess images using parallel processing"""
        images = cls.data_open()  
        with ThreadPoolExecutor() as executor:
            results = executor.map(cls.process_single_image, images)
            for result in results:
                if result is not None:
                    cls.preprocessed_images.append(result)
        
        return cls.preprocessed_images



 

