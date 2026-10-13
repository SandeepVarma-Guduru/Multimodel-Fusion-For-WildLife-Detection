# Multimodal Fusion for Wildlife Detection

## Overview

This project implements a **wildlife approach detection system** using multimodal sensor data by combining Infrared (IR) signals and environmental audio signals. The system extracts frequency-domain features using the Fast Fourier Transform (FFT) and applies machine learning algorithms to classify whether wildlife is approaching or not approaching.

The project explores feature-level fusion, class balancing, dimensionality reduction, model evaluation, and the impact of different frequency resolutions on classification performance.

**Note:** This implementation uses synthetic IR and audio signals for experimentation. Real-world wildlife sensor data is required to validate practical performance.

## Objectives

- Detect approaching and non-approaching wildlife using simulated sensor signals.
- Extract frequency-domain features using FFT.
- Combine IR and audio features through feature-level fusion.
- Handle class imbalance using SMOTE.
- Train and evaluate multiple machine learning models.
- Compare different IR and audio frequency resolutions.
- Evaluate performance using AUC, accuracy, confusion matrices, and cross-validation.

## System Architecture

```text
        Synthetic IR Signal
                |
           FFT Analysis
                |
       IR Feature Extraction
                |
                +------------------+
                                   |
                                   v
                            Feature Fusion
                                   ^
                                   |
                +------------------+
                |
       Audio Feature Extraction
                |
           FFT Analysis
                |
       Synthetic Audio Signal

                  |
                  v
          Fused Feature Dataset
                  |
             Train/Test Split
                  |
          SMOTE Oversampling
                  |
          Machine Learning Models
                  |
          +-------+---------+
          |       |         |
       Random   Logistic   PCA +
       Forest  Regression  Random Forest
          |       |         |
          +-------+---------+
                  |
          Model Evaluation
                  |
       AUC, Accuracy, ROC Curve,
       Cross-Validation and Plots
```

## Technologies Used

- **Programming Language:** Python
- **Numerical Computing:** NumPy
- **Data Processing:** Pandas
- **Signal Processing:** SciPy FFT
- **Machine Learning:** Scikit-learn
- **Class Imbalance Handling:** Imbalanced-learn (SMOTE)
- **Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook / Google Colab

## Methodology

### 1. Synthetic Data Generation

The implementation generates simulated IR and environmental audio signals for two classes:

- `1` — Approaching wildlife
- `0` — Not approaching wildlife

A total of 1,000 samples are generated, consisting of 450 approaching and 550 non-approaching samples.

### 2. FFT-Based Feature Extraction

The Fast Fourier Transform converts signals from the time domain into the frequency domain.

- **IR signals:** Sampled at 10 Hz for 10 seconds.
- **Audio signals:** Sampled at 48 kHz for 2 seconds.

The power spectrum is calculated from the FFT magnitude. Frequency bins are then used to extract features from both modalities.

### 3. Multimodal Feature Fusion

The extracted IR and audio features are concatenated into a single feature vector for each sample. This is known as **feature-level fusion**.

The fused features allow the classifier to use information from both sensor modalities.

### 4. Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

Stratified splitting preserves the class proportions in both sets.

### 5. Class Balancing Using SMOTE

Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic examples of the minority class in the training data to address class imbalance.

SMOTE is applied within the machine learning pipeline to avoid applying oversampling directly to the held-out test set.

### 6. Machine Learning Models

The notebook evaluates the following approaches:

- **Random Forest + SMOTE:** Classification using an ensemble of decision trees.
- **Logistic Regression + SMOTE:** Classification using a linear probabilistic model with feature scaling.
- **PCA + Random Forest + SMOTE:** Dimensionality reduction using Principal Component Analysis before Random Forest classification.

### 7. Model Evaluation

The implementation evaluates model performance using:

- Accuracy
- ROC-AUC score
- Confusion matrix
- Precision, recall, and F1-score
- ROC curve
- Stratified 5-fold cross-validation

### 8. Frequency Resolution Comparison

Six configurations are evaluated using different IR and audio frequency resolutions.

| Configuration | IR Resolution | Audio Resolution |
|---|---:|---:|
| Dataset 1 | 0.1 Hz | 1 kHz |
| Dataset 2 | 0.1 Hz | 2 kHz |
| Dataset 3 | 0.1 Hz | 4 kHz |
| Dataset 4 | 1.0 Hz | 1 kHz |
| Dataset 5 | 1.0 Hz | 2 kHz |
| Dataset 6 | 1.0 Hz | 4 kHz |

The results are compared using AUC and accuracy plots.

## Project Structure

```text
Multimodel-Fusion-For-WildLife-Detection/
│
├── main.ipynb
├── Outputs/
├── README.md
└── wildlife_multimodal_results.csv
```

The CSV file is generated when the notebook completes successfully. The `Outputs/` directory contains project outputs, depending on the files saved there.

## Installation and Execution

### Option 1: Google Colab

1. Open Google Colab.
2. Upload `main.ipynb`.
3. Run the notebook cells.
4. View the model evaluation metrics, plots, and configuration comparison.

### Option 2: Local Jupyter Notebook

Install the required packages:

```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `main.ipynb` and execute the notebook.

The notebook also includes a package installation command for its required libraries.

## Output

After successful execution, the notebook produces:

- Dataset and feature matrix information
- Random Forest classification results
- Logistic Regression results
- PCA + Random Forest results
- Confusion matrix and classification report
- ROC curve and AUC score
- Stratified 5-fold cross-validation results
- Six-configuration comparison table
- AUC comparison plot
- Accuracy comparison plot
- `wildlife_multimodal_results.csv`

Actual numerical results depend on execution and the generated synthetic data. No performance value is assumed in this README.

## Limitations

- The current implementation uses synthetic signals rather than recorded wildlife data.
- The generated signal patterns are simplified and may not represent real animal movement or environmental acoustics.
- High classification performance on synthetic data does not establish real-world detection accuracy.
- Validation using real IR sensor measurements and environmental audio recordings is necessary before practical deployment.

## Future Improvements

- Integrate real-world IR and environmental audio datasets.
- Evaluate the model on recordings from different wildlife species and environments.
- Add noise robustness testing and sensor calibration.
- Compare additional feature extraction and classification techniques.
- Optimize the model for deployment on embedded or edge devices.
- Evaluate false alarm rates and detection latency in real-world conditions.

## Conclusion

This project demonstrates a multimodal machine learning workflow for wildlife approach detection by combining IR and environmental audio features. It integrates FFT-based feature extraction, feature-level fusion, SMOTE, Random Forest, Logistic Regression, PCA, and multiple evaluation techniques.

The implementation provides an experimental foundation for further research using real-world wildlife sensor data.

## Author

**Sandeep Varma Guduru**

- GitHub: https://github.com/SandeepVarma-Guduru
