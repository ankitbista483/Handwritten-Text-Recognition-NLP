import cv2 as cv
import numpy as np
import random
from src.component.data_cleaning import DataCleaner

class DataTransformer:
    @staticmethod
    def deskew(image):
        """Correct skew in handwritten text."""
        coords = np.column_stack(np.where(image > 0))
        angle = cv.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        
        return deskewed

    @staticmethod
    def apply_morphological_operations(image):
        """Reduce noise using morphological operations."""
        kernel = np.ones((3, 3), np.uint8)
        cleaned_image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
        return cleaned_image

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
        """Apply transformations to preprocessed images."""
        preprocessed_images = DataCleaner.preprocess_images_parallel()
        transformed_images = []

        for img in preprocessed_images:
            deskewed = cls.deskew(img)
            cleaned = cls.apply_morphological_operations(deskewed)
            augmented = cls.augment_image(cleaned)
            transformed_images.append(augmented)

        return transformed_images




