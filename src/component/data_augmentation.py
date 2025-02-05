# data_transformation.py

from src.component.data_cleaning import DataCleaner


transformed_images = DataCleaner.preprocess_images_parallel()
print(len(transformed_images))