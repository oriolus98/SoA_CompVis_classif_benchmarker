"""
    Load all trained models, predict class for each test image, aggregate all predictions

    Compute accuracy, mean precission and recall for each aggregate metric
    Available metrics: ['min', 'max','mean', 'sugeno', 'choquet','owa']

"""

import numpy as np
import os
import logging
from datetime import datetime as dt
import tensorflow as tf
import tensorflow.keras.applications as apps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from utils.tools import AggregatePredictions
import pandas as pd
import json
from tqdm import tqdm
import importlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



# Directory containing the images
images_dir = 'data/test'
log_path = 'results/logs/aggregate_metrics'
models_path = 'results'
framework = 'tensorflow'
metrics = ['min', 'max','mean', 'sugeno', 'choquet','OWA_much', 'OWA_least', 'OWA_maj']

if not os.path.isdir(log_path):
    os.makedirs(log_path)

filename = dt.today().strftime('%y%m%d')  + '.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',

    handlers=[
        logging.FileHandler(os.path.join(log_path, filename)),
        logging.StreamHandler()
    ]
)



def plot_conf_matrix(true_classes, predicted_classes, agg_function, class_labels):
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    # Calculate class-wise normalization factor
    row_sums = conf_matrix.sum(axis=1, keepdims=True)

    # Normalize confusion matrix
    normalized_conf_matrix = conf_matrix / row_sums

    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 8))  
    plt.imshow(normalized_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix', fontsize=16)  # Increase title font size
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=90)  # Rotate x-axis labels by 90 degrees
    plt.yticks(tick_marks, class_labels, fontsize=8)  # Adjust y-axis labels font size

    # Add text annotations
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, "{:.2f}".format(normalized_conf_matrix[i, j]), horizontalalignment='center',
                    fontsize=8, color='white' if normalized_conf_matrix[i, j] > 0.5 else 'black')

    plt.xlabel('Predicted Label', fontsize=12)  # Increase x-axis label font size
    plt.ylabel('True Label', fontsize=12)  # Increase y-axis label font size

    plt.tight_layout()
    plt.savefig(os.path.join(log_path, agg_function + 'conf_matrix.png'))
    plt.close()


# Create an ImageDataGenerator
datagen = ImageDataGenerator()

# Create a generator for loading images from the directory
image_generator = datagen.flow_from_directory(
    images_dir,
    target_size=(224, 224),  # Adjust target_size according to your model input size
    batch_size=1,            # Predict one image at a time
    class_mode='categorical',         # One hot encoded labels
    shuffle=False            # Do not shuffle images, so predictions match file order
)

class_labels = image_generator.class_indices

with open('config/model_config.json', 'r') as f:
        config = json.load(f)

models = []
preprocessors = []

for experiment, model_info in config[framework].items():
    input_size = (model_info['input_size'], model_info['input_size'])
    preprocess_path = model_info['preprocess_function']
    # Dynamically import the preprocessor module
    prep_path, class_name = preprocess_path.rsplit('.', 1)
    module = importlib.import_module(prep_path)
    # Get the model class from the module
    preprocessors.append(getattr(module, class_name))
    models.append(load_model(os.path.join(models_path, framework, experiment, 'best_model.h5')))

num_models= len(models) 
num_images = len(image_generator.filenames)
num_classes = len(class_labels)

## We are gonna save all predictions in a cache array so we don't need to recompute for every metric
# Preallocate array for all predictions
all_preds = np.zeros((num_images, num_classes, num_models))

# Fill with TensorFlow model predictions
for i, (model, preprocessor) in tqdm(enumerate(zip(models, preprocessors)), desc="TF Models"):
    image_generator.reset()
    for j, (img, _) in zip(range(num_images), image_generator):
        all_preds[j,:,i] = model.predict(preprocessor(img), verbose=0)[0]

results = {}

for metric in metrics:
    try:
        y_pred = []
        true_labels = []

        # Loop through all images in the directory
        for i, (img, labels) in zip(range(num_images), image_generator):
            true_labels.append(np.argmax(labels))
            pred_batch = all_preds[i]

            agg = AggregatePredictions(metric= metric)
            pred = agg.agg_pred(pred_batch)

            # Determine the class with the highest mean probability
            y_pred.append(np.argmax(pred))


        res = {
                'accuracy': accuracy_score(true_labels, y_pred),
                'precission': precision_score(true_labels, y_pred, average= 'macro'),
                'recall': recall_score(true_labels, y_pred, average= 'macro'),
                'f1': f1_score(true_labels, y_pred, average= 'macro')
        }
        
        results[metric] = res
        plot_conf_matrix(true_labels, y_pred, metric, class_labels)

    except Exception as e:
        logging.error(f"Error in metric {metric}: {e}")
        continue

        
    
df = pd.DataFrame.from_dict(results, orient='index')
df.index.name = 'Aggregation function'
df.reset_index(inplace=True)
df.to_csv(os.path.join(log_path, 'test_agg_metrics.csv'), index=False)
