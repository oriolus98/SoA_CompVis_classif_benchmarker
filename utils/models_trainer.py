import os
import numpy as np
import logging
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


class TrainModels:
    def __init__(self, app_name, feature_extractor, preprocessor, datasets, epochs, batch_size, inp_size, paths, path_to_trained_model = None, do_transfer = False):
        self.train_data_dir = datasets[0]
        self.validation_data_dir = datasets[1]
        self.test_data_dir = datasets[2]
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.input_size = inp_size
        self.log_path = paths[0]
        self.model_path = paths[1]
        self.figures_path = paths[2]
        self.path_to_trained_model = path_to_trained_model
        self.do_transfer = do_transfer

        if self.do_transfer:
            self.app_name = app_name + '_transfer'
        else:
            self.app_name = app_name

        self.logging_conf()


    def train(self):
        self.preprocess_dataset()

        logging.info('Loading model:  {}'.format(self.app_name))
        inputs = layers.Input(shape = (224,224,3))

        if self.do_transfer:
            base_model = self.feature_extractor(weights = 'imagenet', include_top = False, input_tensor = inputs)
            base_model.trainable = False
        else:
            base_model = self.feature_extractor(weights = None, include_top = False, input_tensor = inputs)

        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='swish'),
            layers.Dropout(rate=0.2),
            layers.Dense(self.train_generator.num_classes, activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


        self.check_path = os.path.join(self.model_path, self.app_name + '.h5')
        model_checkpoint = ModelCheckpoint(self.check_path, monitor='val_accuracy', mode='max', save_best_only=True)
        # Train the model
        self.history = model.fit(self.train_generator, epochs=self.num_epochs, validation_data=self.validation_generator, callbacks=[model_checkpoint])

        self.save_learning_curves()
        test_metrics = self.summary_statistics()

        return test_metrics



    def preprocess_dataset(self):
        # Data preprocessing (images must not be previously rescaled, as models preprocessors already implement it)
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            preprocessing_function=self.preprocessor,
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessor,
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='categorical',
        )

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='categorical',
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.test_data_dir,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='categorical',
        )

        logging.info('Dataset preprocessed')


    def save_learning_curves(self):
        # plot and save accuracy and loss evolution
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.num_epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.ylabel("Accuracy (training and validation)")
        plt.xlabel("Training Steps")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.ylabel("Loss (training and validation)")
        plt.xlabel("Training Steps")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, self.app_name + '_acc_and_val.png'))

        logging.info('Learning curve figures saved')


    def summary_statistics(self):

        # if used separately to summary a trained model, load model and dataset, else used in the hole pipeline in .train()
        if self.path_to_trained_model:
            self.check_path = self.path_to_trained_model
            self.preprocess_dataset()

        # Evaluate the model on the test set
        best_model = load_model(self.check_path)
        test_loss, test_accuracy = best_model.evaluate(self.test_generator)
        logging.info('Test Accuracy: {}'.format(test_accuracy))

        # classification repport
        y_true = self.test_generator.classes


        predictions = best_model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        cr = classification_report(y_true, predicted_classes, target_names= self.test_generator.class_indices.keys())

        metrics = {}
        class_metrics = []

        for class_name, class_dict in cr.items():
            if isinstance(class_dict, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics[class_name] = {
                    'precision': class_dict['precision'],
                    'recall': class_dict['recall'],
                    'f1-score': class_dict['f1-score']
                }
                class_metrics.append(metrics[class_name])

        macro_avg = {
            'accuracy': test_accuracy,
            'precision': sum(m['precision'] for m in class_metrics) / len(class_metrics),
            'recall': sum(m['recall'] for m in class_metrics) / len(class_metrics),
            'f1-score': sum(m['f1-score'] for m in class_metrics) / len(class_metrics),
            'transfer learning': self.do_transfer,
            'epochs': self.num_epochs
        }
        
        # Confussion matrix
        true_classes = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())

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
        plt.savefig(os.path.join(self.figures_path, self.app_name + '_conf_matrix.png'))

        return macro_avg


    def logging_conf(self):
        if not os.path.isdir(os.path.join(self.log_path, self.app_name)):
            os.mkdir(os.path.join(self.log_path, self.app_name))

        filename = dt.today().strftime('%y%m%d')  + '.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(asctime)s - %(message)s',

            handlers=[
                logging.FileHandler(os.path.join(self.log_path, self.app_name, filename)),
                logging.StreamHandler()
            ]
        )