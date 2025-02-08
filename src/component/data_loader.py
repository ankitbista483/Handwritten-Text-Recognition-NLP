import os


class DataLoader:
    def __init__(self):
        self.images = []

    def image_loader(self):
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
    

  


