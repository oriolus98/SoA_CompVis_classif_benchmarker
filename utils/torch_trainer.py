import os
import numpy as np
import logging
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class TopLayers(nn.Module):
    """
        A custom neural network head for classification tasks.

        This module defines a simple fully connected neural network head with two layers, 
        intended to be used on top of a feature extractor. It includes a linear layer, 
        followed by a Swish activation, dropout regularization, and a final linear layer 
        for classification.

        Parameters
        ----------
        input_size : int, optional
            The size of the input features (default is 576).
        num_classes : int, optional
            The number of classes for the classification task (default is 10).

        Attributes
        ----------
        fc1 : nn.Linear
            The first fully connected layer that reduces input size to 512 units.
        swish : nn.Hardswish
            The Swish activation function applied after the first layer.
        dropout : nn.Dropout
            Dropout layer with a probability of 0.2 to prevent overfitting.
        fc2 : nn.Linear
            The final fully connected layer that outputs class scores.
    """
    def __init__(self, input_size = 576, num_classes = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.swish = nn.Hardswish()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
            Defines the forward pass through the network.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor with shape (batch_size, input_size).

            Returns
            -------
            x: torch.Tensor
                Output of the classifier
        """
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return(x)

        

class TrainModelsTorch:
    """
        A class to train pyTorch models using transfer learning and standard training configurations.
        It receives a given architecture and a choice of hyperparameters and produces a training and performance
        repport and saves the best performing model.

        Attributes
        ----------
        train_data_dir : str
            Directory path to the training dataset.
        validation_data_dir : str
            Directory path to the validation dataset.
        test_data_dir : str
            Directory path to the test dataset.
        feature_extractor : function
            Base feature extractor model.
        num_epochs : int
            Number of epochs for training.
        batch_size : int
            Batch size for data loading.
        input_size : tuple
            Dimensions (width, height) for resizing images.
        results_path : str
            Path to save results, models, and logs.
        lr : float, optional
            Learning rate for the optimizer (default is 0.001).
        do_transfer : bool, optional
            Indicates whether to use transfer learning (default is True).
        hid_size : int
            Output size of the feature_extractor base model, needed to define input size of classification layer (default is 576)
        app_name : str
            Name of the application/model for experiment tracking.

        Methods
        -------
        train():
            Compiles, trains, and evaluates the model, saving the best model and learning curves.
        preprocess_dataset():
            Prepares the dataloaders for training, validation, and testing.
        save_learning_curves():
            Plots and saves training and validation accuracy/loss curves.
        summary_statistics():
            Evaluates the trained model on the test dataset, returning test metrics and confusion matrix.
        logging_conf():
            Configures logging for experiment tracking.
    """

    def __init__(
        self, 
        app_name: str, 
        feature_extractor: callable, 
        datasets: list[str], 
        epochs: int, 
        batch_size: int, 
        inp_size: tuple[int, int], 
        results_path: str, 
        hid_size: int = 576,
        learning_rate: float = 0.001,  
        do_transfer: bool = True
    ):
        self.app_name = app_name
        self.train_data_dir = datasets[0]
        self.validation_data_dir = datasets[1]
        self.test_data_dir = datasets[2]
        self.feature_extractor = feature_extractor
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.input_size = inp_size
        self.results_path = results_path
        self.hid_size = hid_size
        self.lr = learning_rate
        self.do_transfer = do_transfer
        self.device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )

        self.logging_conf()


    def train(self):
        """
            Preprocesses the dataset, compiles, trains, and evaluates the model. Saves the best model
            and plots the training curves.

            Returns
            -------
            dict
                Test metrics such as accuracy, precision, recall, and F1 score, along used model and hyperparameters.
        """

        logging.info('Experiment:  {}'.format(self.app_name))
        logging.info('Loading model:  {}'.format(self.feature_extractor.__name__))
        self.preprocess_dataset()

        if self.do_transfer:
            base_model = self.feature_extractor(weights='IMAGENET1K_V1').to(self.device)
            for param in base_model.parameters():
                param.requires_grad = False
        else:
            base_model = self.feature_extractor(weights=None).to(self.device)            

        base_model.classifier = TopLayers(input_size = self.hid_size, num_classes = self.num_classes).to(self.device)
        base_model = base_model.to(self.device)

        self.metric = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params= base_model.parameters(), lr= self.lr, weight_decay=0.0001)

        best_accuracy = 0
        self.check_path = os.path.join(self.results_path, self.app_name, 'best_model.pth')

        test_loss = []
        test_accuracy = []
        train_loss = []
        train_accuracy= []

        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch+1}\n-------------------------------")
            loss, acc = self.train_step(base_model, optimizer)
            train_loss.append(loss)
            train_accuracy.append(acc)
            loss, acc = self.test_step(self.validDL, base_model)
            test_loss.append(loss)
            test_accuracy.append(acc)

            if acc > best_accuracy:
                best_accuracy = acc
                logging.info('Saving checkpoint')
                torch.save(base_model.state_dict(), self.check_path)


        self.save_learning_curves(train_accuracy, train_loss, test_accuracy, test_loss)
        test_metrics = self.summary_statistics()

        return test_metrics


    def preprocess_dataset(self):
        """
            Initializes dataloaders for training, validation, and testing datasets, adds random data augmentation
        """
        trf_train = transforms.Compose([transforms.Resize(self.input_size),
                        transforms.RandomAffine(degrees=0, shear=0.2),  
                        transforms.RandomResizedCrop(size=self.input_size, scale=(0.8, 1.0)),  
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        trf_test = transforms.Compose([transforms.Resize(self.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

        train_dataset = ImageFolder(root = self.train_data_dir, transform= trf_train)
        test_dataset = ImageFolder(root = self.test_data_dir, transform= trf_test)
        valid_dataset = ImageFolder(root = self.validation_data_dir, transform= trf_test)

        self.trainDL = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.testDL = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        self.validDL = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        self.num_classes = len(train_dataset.classes)
        self.class_names = train_dataset.classes

        logging.info('Dataset preprocessed')


    def save_learning_curves(self, train_acc, train_loss, val_acc, val_loss):
        """
            Plots and saves the training and validation accuracy and loss curves to the results directory.
        """
        epochs_range = range(self.num_epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.ylabel("Accuracy (training and validation)")
        plt.xlabel("Training Steps")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.ylabel("Loss (training and validation)")
        plt.xlabel("Training Steps")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, self.app_name, 'training_curves.png'))

        logging.info('Learning curve figures saved')


    def summary_statistics(self, path_to_trained_model: str = None):
        """
            Evaluates the model on the test dataset, calculates test metrics and confusion matrix, and saves them.

            Parameters
            -------
            path_to_trained_model : str, optional
                Path to a pre-trained model for further evaluation (default is None).

            Returns
            -------
            dict
                Test accuracy, precision, recall, and F1 score, along with additional model info.
        """
        # if used separately to summary a trained model, load model and dataset, else used in the hole pipeline in .train()
        if path_to_trained_model:
            self.check_path = path_to_trained_model
            self.preprocess_dataset()

        # Evaluate the model on the test set
        best_model = self.feature_extractor(weights = None).to(self.device)
        best_model.classifier = TopLayers(input_size = self.hid_size, num_classes = self.num_classes).to(self.device)
        best_model = best_model.to(self.device)
        best_model.load_state_dict(torch.load(self.check_path, weights_only=True))
        best_model = best_model.to(self.device)

        test_loss, test_accuracy = self.test_step(self.testDL, best_model)
        logging.info('Test Accuracy: {}'.format(test_accuracy))

        all_preds = []
        all_labels = []

        best_model.eval()
        with torch.no_grad():
            for X, y in self.testDL:
                X, y = X.to(self.device), y.to(self.device)
                predictions = best_model(X)
                _, predicted_classes = torch.max(predictions, 1)
                all_preds.extend(predicted_classes.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        cr = classification_report(all_labels, all_preds, target_names= self.class_names, output_dict=True)

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
            'model': self.feature_extractor.__name__,
            'accuracy': test_accuracy,
            'precision': sum(m['precision'] for m in class_metrics) / len(class_metrics),
            'recall': sum(m['recall'] for m in class_metrics) / len(class_metrics),
            'f1-score': sum(m['f1-score'] for m in class_metrics) / len(class_metrics),
            'transfer learning': self.do_transfer,
            'epochs': self.num_epochs,
            'lr': self.lr,
            'batch size': self.batch_size
        }
        
        # Confussion matrix
        class_labels = self.class_names

        conf_matrix = confusion_matrix(all_labels, all_preds)

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
        plt.savefig(os.path.join(self.results_path, self.app_name, + 'conf_matrix.png'))

        return macro_avg


    def train_step(self, model, optimizer):
        """
            Performs a single training step (forward and backward) over the entire training dataset.

            Parameters
            ----------
            model : torch.nn.Module
                The neural network model to train.
            optimizer : torch.optim.Optimizer
                The optimizer used to update model parameters.

            Returns
            -------
            avg_train_loss : float
                The average training loss over the dataset.
            train_accuracy : float
                The training accuracy as a ratio of correct predictions to total samples.
        """
        size = len(self.trainDL.dataset)
        total_loss, correct = 0, 0
        model.train()
        for batch, (X, y) in enumerate(self.trainDL):
            X, y = X.to(self.device), y.to(self.device)

            pred = model(X)
            loss = self.metric(pred, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * X.size(0) 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 30 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        avg_train_loss = total_loss / size
        train_accuracy = correct / size
        return avg_train_loss, train_accuracy


    def test_step(self, dataloader, model):
        """
            Evaluates the model's performance on a given dataset without updating weights.

            Parameters
            ----------
            dataloader : torch.utils.data.DataLoader
                The DataLoader providing test data batches.
            model : torch.nn.Module
                The neural network model to evaluate.

            Returns
            -------
            test_loss : float
                The average test loss over the entire dataset.
            correct : float
                The accuracy on the test dataset as a ratio of correct predictions to total samples.
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        model.eval()

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += self.metric(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        logging.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, correct


    def logging_conf(self):
        """
            Configures logging for experiment tracking.
        """
        if not os.path.isdir(os.path.join(self.results_path, self.app_name)):
            os.makedirs(os.path.join(self.results_path, self.app_name))

        filename = dt.today().strftime('%y%m%d')  + '.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(asctime)s - %(message)s',

            handlers=[
                logging.FileHandler(os.path.join(self.results_path, self.app_name, filename)),
                logging.StreamHandler()
            ]
        )