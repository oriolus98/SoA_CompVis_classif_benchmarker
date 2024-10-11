# Custom image classification benchmarker

When dealing with an image classification new problem, sometimes we need to compare which State of the Art model performs better for this given task. This simple application automates the process of benchmarking and comparing several models using tensorflow. 

The images dataset must be divided into train, test and validation subfolders within './dataset'. All Computer vision models must be defined in './model_config.json', including number of epochs, if transfer learning should be applied etc. Selected hyperparameters, along different obtained metrics on the validation dataset for each model are saved on './results/test_metrics.csv'.

Furthermore, aggregating the results of different trained models through the use of aggregation functions has been shown to increase the reliability and robustness of image classification predictions. Therefore, once the main benchmark has been launched and different models are trained for our current dataset, we can use './agg_pred.py' to evaluate and benchmark different aggreagation functions for the proposed dataset. 

