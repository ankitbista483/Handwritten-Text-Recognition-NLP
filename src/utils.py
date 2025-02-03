import os 

class DataCleaner:
    def remove_json(self):
        """ This function is used to clean the json file that was inside the dataset
        """
        use_dir = 'dataset'
        files = os.listdir(use_dir)
        for file in files:
            if file == '.DS_Store':
                continue
            file_path = os.path.join(use_dir,file)
            image_path = os.listdir(file_path)
            for image_name in image_path:
                image = os.path.join(file_path, image_name)
                if image.endswith('json'):
                    os.remove(image)
    

      