import os
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime as dt
import importlib
from utils.tools import CheckImages



def main():
    framework = 'torch'

    results_path = os.path.join('results',framework)

    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Define paths to images dataset
    train_data_dir = 'dataset/train'
    validation_data_dir = 'dataset/valid'
    test_data_dir = 'dataset/test'

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

    for experiment, model_info in config[framework].items():
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
            from utils.models_trainer import TrainModelsTF
            preprocess_path = model_info['preprocess_function']
            # Dynamically import the preprocessor module
            prep_path, class_name = preprocess_path.rsplit('.', 1)
            module = importlib.import_module(prep_path)
            # Get the model class from the module
            preprocessor_class = getattr(module, class_name)

            mod = TrainModelsTF(app_name = experiment, feature_extractor = model_class, preprocessor = preprocessor_class, datasets = datasets, epochs = n_epochs, batch_size = batch_size, inp_size = input_size, learning_rate = lr, results_path = results_path, do_transfer = do_transfer)
            results[experiment] = mod.train()

        elif framework == 'torch':
            from utils.torch_trainer import TrainModelsTorch
            preprocessor_class = None
            hid_size = model_info['hidden_size']

            mod = TrainModelsTorch(app_name = experiment, feature_extractor = model_class, datasets = datasets, epochs = n_epochs, batch_size = batch_size, inp_size = input_size, hid_size = hid_size, learning_rate = lr, results_path = results_path, do_transfer = do_transfer)
            results[experiment] = mod.train()
        
    
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'Experiment'
        df.reset_index(inplace=True)
        df.to_csv(os.path.join(results_path, 'test_metrics.csv'), index=False)


if __name__ == "__main__":
    main()