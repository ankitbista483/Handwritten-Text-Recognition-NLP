import cv2 as cv

from src.component.data_cleaning import DataCleaner

from src.component.data_cleaning import DataCleaner

\
from src.component.data_cleaning import DataCleaner, DataLoader

# Create DataLoader instance
data_loader = DataLoader()
images = data_loader.data_open()

# Now pass images to DataCleaner
data_cleaner = DataCleaner(images)


# Load and preprocess images

data_cleaner.preprocess_images_parallel()

# Now you can access the preprocessed images
preprocessed_images = data_cleaner.preprocessed_images

# You can now use these preprocessed images
print(f"Number of preprocessed images: {len(preprocessed_images)}")
