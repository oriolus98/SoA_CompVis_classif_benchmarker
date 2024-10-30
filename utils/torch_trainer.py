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
    def __init__(self, input_size = 576, num_classes = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.swish = nn.Hardswish()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return(x)

        

class TrainModelsTorch:
    def __init__(self, app_name, feature_extractor, datasets, epochs, batch_size, inp_size, results_path, hid_size = 576, learning_rate = 0.001, path_to_trained_model = None, do_transfer = False):
        self.train_data_dir = datasets[0]
        self.validation_data_dir = datasets[1]
        self.test_data_dir = datasets[2]
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.input_size = inp_size
        self.results_path = results_path
        self.hid_size = hid_size
        self.lr = learning_rate
        self.path_to_trained_model = path_to_trained_model
        self.do_transfer = do_transfer
        self.device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )

        if self.do_transfer:
            self.app_name = app_name + '_transfer'
        else:
            self.app_name = app_name

        self.logging_conf()


    def train(self):

        self.preprocess_dataset()

        logging.info('Experiment:  {}'.format(self.app_name))
        logging.info('Loading model:  {}'.format(self.feature_extractor.__name__))

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
            loss, acc = self.test_step(self.validDL, base_model, self.metric)
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

        self.trainDL = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.testDL = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        self.validDL = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        self.num_classes = len(train_dataset.classes)
        self.class_names = train_dataset.classes

        logging.info('Dataset preprocessed')


    def save_learning_curves(self, train_acc, train_loss, val_acc, val_loss):

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


    def summary_statistics(self):

        # if used separately to summary a trained model, load model and dataset, else used in the hole pipeline in .train()
        if self.path_to_trained_model:
            self.check_path = self.path_to_trained_model
            self.preprocess_dataset()

        # Evaluate the model on the test set
        best_model = self.feature_extractor(weights = None).to(self.device)
        best_model.classifier = TopLayers(input_size = self.hid_size, num_classes = self.num_classes).to(self.device)
        best_model = base_model.to(self.device)
        best_model.load_state_dict(torch.load(self.check_path, weights_only=True))
        best_model = final_model.to(self.device)

        test_loss, test_accuracy = self.test_step(self.testDL, best_model, self.metric)
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
        cr = classification_report(all_labels, all_preds, target_names= self.class_names)

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


    def test(self, dataloader, model):
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