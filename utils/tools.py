import PIL
from PIL import Image, ImageDraw
import os
import logging
import numpy as np



class CheckImages:
    def __init__(self, folder_paths):
        self.folder_paths = folder_paths

    def count_corrupt_images(self):
        i = 0
        for folder_path in self.folder_paths:
            for subfolder in os.listdir(folder_path):
                for filename in os.listdir(os.path.join(folder_path, subfolder)):
                    try:
                        image = Image.open(os.path.join(folder_path, subfolder, filename))
                    except PIL.UnidentifiedImageError as e:
                        i = i + 1
                        logging.info(f"Error in file {filename}: {e}")

        return i
    

    def delete_corrupt_images(self):
        i = 0
        for folder_path in self.folder_paths:
            for subfolder in os.listdir(folder_path):
                for filename in os.listdir(os.path.join(folder_path, subfolder)):
                    try:
                        image = Image.open(os.path.join(folder_path, subfolder, filename))
                    except PIL.UnidentifiedImageError as e:
                        i = i+1
                        logging.info(f"Error in file {filename}: {e}")
                        os.remove(os.path.join(folder_path, subfolder, filename))
                        logging.info(f"Removed file {filename}")

        logging.info(f'{i} corrupt images removed')


    def count_images_per_class(self):
        for directory in self.folder_paths:
            print(directory)
            for clas in os.listdir(directory):
                print(clas)
                print(len(os.listdir(os.path.join(directory, clas))))



class ShadowAugmentor:
    """
        Adds random elliptical, rectangular and triangle shadows to a fraction of the dataset

        THIS CLASS IS NOT TESTED YET
    """
    def __init__(self, input_dir, output_dir, n_shadows = 5, p = 0.5):
        self.n_shadows = n_shadows
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fraction = p


    def process_dataset(self):
        """
        Process the entire dataset by adding shadows to each image.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        image_files = os.listdir(self.input_dir)
        num_images = len(image_files)
        num_images_to_process = int(self.fraction * num_images)
        selected_files = np.random.choice(image_files, num_images_to_process, replace=False)

        for filename in selected_files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
                image_path = os.path.join(self.input_dir, filename)
                image = Image.open(image_path)
                self.add_shadows(image, image_path)



    def add_shadows(self, image, image_path):
        tr_shadows = np.random.randint(0, self.n_shadows)
        rec_shadows = np.random.randint(0, self.n_shadows)
        elip_shadows = np.random.randint(0, self.n_shadows)

        shaded_image = image

        for i in range(tr_shadows):
            shaded_image = self.add_triangular_shadow(shaded_image)
        
        for i in range(rec_shadows):
            shaded_image = self.add_rectangular_shadow(shaded_image)
        
        for i in range(tr_shadows):
            shaded_image = self.add_elliptical_shadow(shaded_image)
        

        filename = os.path.splitext(os.path.basename(image_path))[0]
        # Save the image with shadow
        output_path = os.path.join(self.output_dir, f"{filename}_shadow.png")
        shaded_image.save(output_path)
        # return shaded_image
    

    @staticmethod
    def add_rectangular_shadow(image):
        """
        Add rectangula shadow to an image.
        """
        width, height = 225, 225
        
        shadow = Image.new('RGBA', (width, height))

        # Generate random shadow coordinates
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(x1, width), np.random.randint(y1, height)
        
        # Generate random shadow intensity
        intensity = int(np.random.randint(50,150))


        for x in range(x1, x2):
            for y in range(y1, y2):
                # Define the color for the pixel (x, y)
                color = (0, 0, 0, intensity)
                shadow.putpixel((x, y), color) # mask

        image_with_shadow = Image.alpha_composite(image.convert('RGBA'), shadow)
        return image_with_shadow.convert('RGB')
    

    @staticmethod
    def add_elliptical_shadow(image):
        """
        Add elliptical shadow to an image.
        """
        width, height = 225, 225
        
        shadow = Image.new('RGBA', (width, height))

        # Generate random shadow parameters
        x_center, y_center = np.random.randint(0, width), np.random.randint(0, height)
        a = np.random.randint(20, 60)  # major axis
        b = np.random.randint(10, 30)   # minor axis

        # Generate random shadow intensity
        intensity = int(np.random.randint(50, 150))

        # Create an elliptical mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((x_center - a, y_center - b, x_center + a, y_center + b), fill=intensity)

        # Paste the shadow onto the shadow image
        shadow.putalpha(mask)

        image_with_shadow = Image.alpha_composite(image.convert('RGBA'), shadow)
        return image_with_shadow.convert('RGB')
    

    @staticmethod
    def add_triangular_shadow(image):
        """
        Add triangular shadow to an image.
        """
        width, height = 225, 225
        
        shadow = Image.new('RGBA', (width, height))

        # Generate random shadow parameters
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        x3, y3 = np.random.randint(0, width), np.random.randint(0, height)

        # Generate random shadow intensity
        intensity = int(np.random.randint(50, 150))

        # Create a triangular mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=intensity)

        # Paste the shadow onto the shadow image
        shadow.putalpha(mask)

        image_with_shadow = Image.alpha_composite(image.convert('RGBA'), shadow)
        return image_with_shadow.convert('RGB')

    


class AggregatePredictions:
    """
        @author: Cédric Marco Detchart (VRAIN).
        Aggregate predictions from different models into a single predictions array with a single probability for each class 
        Implemented metrics are: ["min", "max", "mean", "choquet", "sugeno", "owa"]
    """
    def __init__(self, metric = 'mean'):
        self.agg = metric


    def agg_pred(self, pred_batch):
        img_batch = pred_batch.shape[1]
        
        if (self.agg == 'min'):
            pred = np.min(pred_batch, axis=1)
        elif (self.agg == 'max'):
            pred = np.max(pred_batch, axis=1)
        elif (self.agg == 'mean'):
            pred = np.mean(pred_batch, axis=1)
        elif (self.agg == 'choquet'):
            fm = np.array(range(img_batch-1, 0, -1)) / img_batch
            pred_batch = np.sort(pred_batch,axis=1)
            pred = pred_batch[:,0] + np.sum((pred_batch[:,1:] - pred_batch[:,:-1]) * fm, axis = 1)
        elif (self.agg == 'sugeno'):
            fm = np.array(range(img_batch, 0, -1)) / img_batch
            pred_batch = np.sort(pred_batch,axis=1)
            pred = np.max(np.minimum(pred_batch, fm), axis=1)
        elif (self.agg == 'owa'):
            weights = self.owa_weights(img_batch, a=0.5, b=1)
            pred = self.owa(pred_batch, weights, axis=1)
        else:
            print('Please introduce a valid aggregate metric from: ["min", "max", "mean", "choquet", "sugeno", "owa"]')

        return pred
    


    def owa_weights(self, n, a=None, b=None):

        if (a is not None) and (b is not None):
            # idx = np.array(range(0, n + 1))
            idx = self.quantifier(np.array(range(0, n + 1)) / n, a, b)
            weights = idx[1:] - idx[:-1]
        else:
            weights = np.random.random_sample(n)
        weights = np.sort(weights)[::-1]

        return weights



    @staticmethod
    def owa(x, weights, axis=0):
        """
        :param axis:
        :param x: data to aggregate
        :param weights: weights passed in order to aggregate data
        :return: matrix with the aggregated data
        """
        X_sorted = -np.sort(-x, axis=axis)

        return np.sum(X_sorted * weights, axis=axis)
    
    

    @staticmethod
    def quantifier(x, a, b):
        q = (x - a) / (b - a)

        q[x < a] = 0
        q[x > b] = 1

        return q