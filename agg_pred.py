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
from utils.tools import AggregatePredictions


# Directory containing the images
images_dir = 'dataset/test/'
log_path = 'results/logs/aggregate_metrics'

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

true_labels = []
class_labels = ['SIGRE','blue','green','green_point','grey','yellow']
for subdir in class_labels:
    files = os.listdir(os.path.join(images_dir, subdir))
    true_labels.extend([class_labels.index(subdir)] * len(files))

# Load the saved model
model1 = tf.keras.models.load_model('results/models/EfficientNetV2_transfer/EfficientNetV2B1_transfer.h5')
model0 = tf.keras.models.load_model('results/models/EfficientNetV2_transfer/EfficientNetV2B0_transfer.h5')
model2 = tf.keras.models.load_model('results/models/EfficientNetV2_transfer/EfficientNetV2B2_transfer.h5')
model3 = tf.keras.models.load_model('results/models/EfficientNetV2_transfer/EfficientNetV2B3_transfer.h5')
model4 = tf.keras.models.load_model('results/models/EfficientNetV2_transfer/EfficientNetV2M_transfer.h5')
model5 = tf.keras.models.load_model('results/models/EfficientNetV2_transfer/EfficientNetV2L_transfer.h5')
model6 = tf.keras.models.load_model('results/models/EfficientNetV2_transfer/EfficientNetV2S_transfer.h5')
model7 = tf.keras.models.load_model('results/models/MobileNetV3_transfer/MobileNet_V3_Large_transfer.h5')
model8 = tf.keras.models.load_model('results/models/MobileNetV3_transfer/MobileNet_V3_Small_transfer.h5')
model9 = tf.keras.models.load_model('results/models/Xception_transfer/Xception_transfer.h5')
model10 = tf.keras.models.load_model('results/models/InceptionV3_transfer/InceptionV3_transfer.h5')
model11 = tf.keras.models.load_model('results/models/NasNet_transfer/NasNet_Mobile_transfer.h5')
model12 = tf.keras.models.load_model('results/models/NasNet_transfer/NasNet_Large_transfer.h5')

models = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]
preprocessors = [apps.efficientnet.preprocess_input, apps.efficientnet.preprocess_input, apps.efficientnet.preprocess_input, apps.efficientnet.preprocess_input, apps.efficientnet.preprocess_input, apps.efficientnet.preprocess_input, apps.efficientnet.preprocess_input, apps.mobilenet_v3.preprocess_input, apps.mobilenet_v3.preprocess_input, apps.xception.preprocess_input, apps.inception_v3.preprocess_input, apps.nasnet.preprocess_input, apps.nasnet.preprocess_input]


# Create an ImageDataGenerator
datagen = ImageDataGenerator()

# Create a generator for loading images from the directory
image_generator = datagen.flow_from_directory(
    images_dir,
    target_size=(224, 224),  # Adjust target_size according to your model input size
    batch_size=1,            # Predict one image at a time
    class_mode=None,         # No labels are provided
    shuffle=False            # Do not shuffle images, so predictions match file order
)

metrics = ['min', 'max','mean', 'sugeno', 'choquet','owa']

for metric in metrics:
    count = 0
    total = 0

    # Initialize true positive, false positive, and false negative counts for each class
    tp = np.zeros(len(class_labels))
    fp = np.zeros(len(class_labels))
    fn = np.zeros(len(class_labels))

    # Loop through all images in the directory
    for img, img_path in zip(image_generator, image_generator.filenames):

        print("Predictions for image:", img_path)
        
        # Initialize list to store predictions for this image
        image_predictions = []
        total += 1

        # Iterate over all models
        for model, preprocessor in zip(models, preprocessors):
            # Preprocess the image
            preprocessed_img = preprocessor(img)
            
            # Predict using the current model
            prediction = model.predict(preprocessed_img)
            
            # Append the prediction to the list of predictions for this model
            image_predictions.append(prediction.flatten())  # Flatten prediction to (6,) array

        # Convert predictions for this image to numpy array
        image_predictions = np.array(image_predictions)
        pred_batch = image_predictions.transpose()

        agg = AggregatePredictions(metric= metric)
        pred = agg.agg_pred(pred_batch)

        # Determine the class with the highest mean probability
        highest_pred_loc = np.argmax(pred)
        true_label = true_labels[image_generator.batch_index - 1]  # Get true label for the current batch
        predicted_label = highest_pred_loc
        
        if predicted_label == true_label:
            count += 1
            tp[true_label] += 1
        else:
            fp[predicted_label] += 1
            fn[true_label] += 1

        # Compute precision and recall for each class
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        # Compute average precision and recall
        average_precision = np.mean(precision)
        average_recall = np.mean(recall)
        accuracy = count*100/total
    
    logging.info('Aggregation metric: {}'.format(metric))
    logging.info('MoE accuracy: {}%'.format(accuracy))
    logging.info('Average Precision: {}'.format(average_precision))
    logging.info('Average Recall: {}'.format(average_recall))