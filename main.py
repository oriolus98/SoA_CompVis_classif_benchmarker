import os
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime as dt
import importlib
from utils.tools import CheckImages
from utils.models_trainer import TrainModelsTF, TrainModelsTorch



def main():
    framework = 'torch'
    # Configure results directories
    log_path = 'results/logs' 
    model_path = 'results/models'
    figures_path = 'results/figures'

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)


    # Define paths to images dataset
    train_data_dir = 'dataset/train'
    validation_data_dir = 'dataset/validation'
    test_data_dir = 'dataset/test'

    paths = [log_path, model_path, figures_path]
    # check for corrupt images (in the first test some corrupt images at the dataset stopped training)
    datasets = [train_data_dir, validation_data_dir, test_data_dir]

    check = CheckImages(datasets)
    corr_images = check.count_corrupt_images()

    if corr_images:
        raise FileNotFoundError('{} corrupt images found'.format(corr_images))
        exit(1)
    

    with open('config/model_config.json', 'r') as f:
        config = json.load(f)

    results = {}

    for model_name, model_info in config[framework].items():
        model_module_path = model_info['model']
        n_epochs = model_info['num_epochs']
        do_transfer = model_info['transfer_learning']
        batch_size = model_info['batch_size']
        input_size = (model_info['input_size'], model_info['input_size'])
        lr = model_info['learning_rate']

        # Dynamically import the model module
        module_path, class_name = model_module_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        # Get the model class from the module
        model_class = getattr(module, class_name)

        if framework == 'tensorflow':
            preprocess_path = model_info['preprocess_function']
            # Dynamically import the preprocessor module
            prep_path, class_name = preprocess_path.rsplit('.', 1)
            module = importlib.import_module(prep_path)
            # Get the model class from the module
            preprocessor_class = getattr(module, class_name)

            mod = TrainModelsTF(app_name = model_name, feature_extractor = model_class, preprocessor = preprocessor_class, datasets = datasets, epochs = n_epochs, batch_size = batch_size, inp_size = input_size, learning_rate = lr, paths = paths, do_transfer = do_transfer)
            results[model_name] = mod.train()

        elif framework == 'torch':
            preprocessor_class = None

            mod = TrainModelsTorch(app_name = model_name, feature_extractor = model_class, preprocessor = preprocessor_class, datasets = datasets, epochs = n_epochs, batch_size = batch_size, inp_size = input_size, learning_rate = lr, paths = paths, do_transfer = do_transfer)
            results[model_name] = mod.train()
        
    
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'model'
        df.reset_index(inplace=True)
        df.to_csv('./results/test_metrics.csv', index=False)


if __name__ == "__main__":
    main()