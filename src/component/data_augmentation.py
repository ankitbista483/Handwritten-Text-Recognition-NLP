import cv2 as cv
import numpy as np
import random
from src.component.data_cleaning import DataCleaner

class DataTransformer:
    @staticmethod
    def augment_image(image):
        """Apply random rotation, scaling, and brightness adjustments."""
        # Rotation
        angle = random.uniform(-15, 15)
        (h, w) = image.shape[:2]
        M = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv.warpAffine(image, M, (w, h))

        # Scaling
        scale = random.uniform(0.9, 1.1)
        scaled = cv.resize(rotated, None, fx=scale, fy=scale)

        # Brightness Adjustment
        brightness = random.uniform(0.7, 1.3)
        brightened = np.clip(scaled * brightness, 0, 255).astype(np.uint8)

        return brightened

    @classmethod
    def transform_images(cls):
        
        preprocessed_images = DataCleaner.preprocess_images_parallel()
        transformed_images = []

        for img in preprocessed_images:
            augmented = cls.augment_image(img)
            transformed_images.append(augmented)

        return transformed_images




