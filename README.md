# Custom Image Classification Benchmarker

This repository provides a simple and efficient way to benchmark and compare the performance of state-of-the-art image classification models on a custom dataset. It supports both TensorFlow and PyTorch frameworks, making it a versatile tool for experimenting with different deep learning architectures. 

## Features

- **Framework Flexibility**: Choose between TensorFlow and PyTorch by setting `framework = 'tensorflow'` or `framework = 'torch'` in `main.py`.
- **Custom Model Configurations**: Define hyperparameters such as input size, batch size, learning rate, number of epochs, and whether to use pretrained weights in `config/model_config.json`.
- **Model Support**: Automatically supports all models from:
  - TensorFlow: `tensorflow.keras.applications`
  - PyTorch: `torchvision.models`
- **Aggregation Functions**: Evaluate and benchmark different aggregation functions to combine predictions from multiple models for robust classification.
- **GPU Support**: Leverages GPU acceleration for faster training and evaluation.
- **Comprehensive Results**: Saves training curves, confusion matrices, and evaluation metrics for each model.
- **Cross-Framework Consistency**: Unified configuration for models in both TensorFlow and PyTorch.



## Dataset Structure

Your dataset should follow this directory structure:

```bash
.
├── README.md
├── dataset
│   ├── train
│   │   ├── class1/
│   │   ├── class2/
│   │   └── ...
│   ├── test
│   │   ├── class1/
│   │   └── ...
│   └── validation
│       ├── class1/
│       └── ...
└── ...
```

Replace `class1`, `class2`, etc., with the actual class labels of your dataset.



## Usage

### 1. Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/oriolus98/SoA_CompVis_classif_benchmarker.git
   cd SoA_CompVis_classif_benchmarker
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Configuration

Edit `config/model_config.json` to specify the models, frameworks, and hyperparameters. Example:

```json
{
  "torch": {
        "Experiment1": {
            "model": "torchvision.models.efficientnet_v2_l",
            "num_epochs": 12,
            "input_size": 224,
            "batch_size": 64,
            "learning_rate": 0.001,
            "hidden_size": 1280,
            "transfer_learning": true
        }
  },
  "tensorflow": {
       "Experiment4": {
            "model": "tensorflow.keras.applications.MobileNetV3Small",
            "preprocess_function": "tensorflow.keras.applications.mobilenet_v3.preprocess_input",
            "num_epochs": 8,
            "input_size": 224,
            "batch_size": 64,
            "learning_rate": 0.001,
            "transfer_learning": true
       }
  }
}
```

- **TensorFlow Models**: Automatically determines the output size.
- **PyTorch Models**: Specify `hidden_size` for the last layer's input size.

### 3. Running the Benchmark
Run `main.py` to start the benchmarking process:

```bash
   python main.py
```

### 4. Aggregation of Predictions
Once the models are trained, evaluate aggregation functions by running:

```bash
   python agg_pred.py
```

This generates `results/aggregate_benchmark.csv` with the aggregated evaluation metrics.
For the moment, supported aggregation functions include:
- Minimum
- Maximum
- Mean
- Choquet Integral
- Sugeno Integral
- Ordered Weights Averaging (OWA)
  
These functions combine the output from multiple trained models to improve robustness and reliability in classification tasks. For a deeper dive into aggregation functions, see the [Further Reading](#further-reading) section.

---

## Example usage 
Here is an example workflow:

1. Place your dataset in the `dataset/` directory following the required structure.
2. Configure the models and hyperparameters in `config/model_config.json`.
3. Run the benchmark with:
```bash
   python main.py
```
4. Evaluate aggregation functions with:
```bash
   python agg_pred.py
```
5. Review the results in the results/ folder.



## Results
- **Metrics**: Saved in `results/test_metrics.csv`.
- **Metrics from models aggregation**: Saved in `results/aggregate_benchmark.csv`.
- **Training Curves**: Plots saved in `results/`.
- **Confusion Matrices**: Saved in `results/`.
- **Trained models**: Saved in `models/`.



## Requirements

- Python 3.9+
- TensorFlow 2.x
- PyTorch 2.x
- GPU (recommended for faster training)

Install all dependencies using the provided `requirements.txt` file.



## Contributing

This project is open for contributions! Here are some possible areas for improvement:
- **Analyze Computational Performance**: Add training and inference time, as well as model size for each model to the final benchmark, in order to compare all relevant factors.
- **Expand Automation**: Add support for more configurable hyperparameters, such as different optimizers (beyond the default Adam optimizer) and advanced data augmentation options.
- **Improve Aggregation Functionality**: Automate agg_pred.py fully, as it currently requires customization based on the models used in the configuration.
- **Enhance Modularity**: Implement a base model class for the trainers to improve modularity. This could unify the existing TorchTrainer and TFTrainer classes.
Feel free to fork this repository, submit pull requests, or open issues to improve this tool.



## Further Reading

If you are interested in the theoretical background and additional references regarding aggregation functions in this context, please refer to our recent publication:
- O. Chacón-Albero et al. *Towards Sustainable Recycling: Advancements in AI-Based Waste Classification*. In: Highlights in Practical Applications of Agents, Multi-Agent Systems, and Digital Twins 2024. Communications in Computer and Information Science, vol 2149. Springer, Cham. https://doi.org/10.1007/978-3-031-73058-0_2
  
where a simplified version of this repository was used to compute the presented results. If you find this work useful in your research, please consider citing our paper:
```
@inbook{inbook,
author = {Chacón-Albero, Oriol and Campos-Mocholí, M. and Marco-Detchart, Cédric and Julian, V. and Rincon, J.A. and Botti, V.},
year = {2025},
title = {Towards Sustainable Recycling: Advancements in AI-Based Waste Classification},
doi = {10.1007/978-3-031-73058-0_2}
}
```


