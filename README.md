# tremors_prediction
This repository offers a framework for analyzing and predicting tremors using time series data, integrating theoretical foundations, feature extraction, trend analysis, and predictive models. It combines statistical and data-driven methods to support researchers in clinical and computational fields in systematically studying tremor progression.
This repo was created by Yehia Gewily in March, 2025.



# Tremor Prediction and Synthesis Framework

This repository provides a comprehensive framework for analyzing and predicting tremor patterns using time series data. It integrates statistical modeling, machine learning, feature extraction, trend analysis, and data-driven resampling techniques to improve the accuracy of tremor progression predictions.

## Overview

Tremor disorders, such as Parkinsonian and essential tremors, require accurate forecasting for better clinical understanding and treatment planning. This project aims to:

- Extract key statistical and frequency-based features from tremor data.
- Analyze trends in tremor progression.
- Predict future tremor characteristics using statistical and machine learning models.
- Generate synthetic tremor recordings that preserve statistical properties and realistic patterns.
- Provide visualization tools to compare historical and predicted tremor data.

## Methodology

The framework consists of the following core components:

1. **Data Preprocessing**
   - Normalization and smoothing
   - Handling missing values
   - Time series segmentation

2. **Feature Extraction**
   - Time-domain and frequency-domain features
   - Statistical and entropy-based measures
   - Wavelet transform for signal analysis

3. **Trend Analysis**
   - Identifying short-term and long-term tremor trends
   - Detecting periodic patterns and fluctuations

4. **Predictive Modeling**
   - Random Forest regression for trend forecasting
   - Forced trend adjustment for clinical pattern alignment
   - Ensemble and statistical methods to improve accuracy

5. **Bootstrap-Based Resampling**
   - Synthetic data generation for small datasets
   - Confidence interval estimation for predictions

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/tremor-prediction.git
cd tremor-prediction
pip install -r requirements.txt
