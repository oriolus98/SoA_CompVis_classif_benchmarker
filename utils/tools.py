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



class AggregatePredictions:
    """
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
    



