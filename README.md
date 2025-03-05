# Bitcoin Price Prediction using CNN and Transfer Learning

This repository contains the Python code for predicting Bitcoin price movements using Convolutional Neural Networks (CNN) with transfer learning.

## About

This project implements a CNN-based approach for predicting Bitcoin price movements using candlestick chart images. The model uses transfer learning with Google's Inception V3 architecture pre-trained on ImageNet.

## Environment and Installation

The code is organized in Jupyter notebooks and runs on Python with TensorFlow. The main requirements are:

- TensorFlow
- Pandas
- Matplotlib
- mplfinance
- Requests (for data download)

The notebooks are designed to run in any Python environment with the required packages installed.

## Data Collection and Preparation

- `downLoadBtcDayDate.ipynb`: Downloads historical Bitcoin price data from Binance API and saves it to CSV files
- `Candle_stick2.ipynb`: Creates candlestick chart images from the price data

## Model Training and Prediction

- `Candle_CNN.ipynb`: Implements the CNN model using transfer learning with Inception V3
  - Uses pre-trained Inception V3 model with additional custom layers
  - Trains only the additional layers while keeping original Inception V3 weights frozen
  - Predicts price movement direction (up/down/flat)

## Model Architecture

- Base model: Inception V3 pre-trained on ImageNet
- Additional layers added for price movement classification
- Training performed only on the additional layers while keeping base model frozen
- Input: Candlestick chart images
- Output: Price movement prediction (3 classes - up/down/flat)

## Results

The model achieves the following accuracy:

- One day ahead: ~0.45 test accuracy
- Three days ahead: ~0.47 test accuracy

## Usage

1. Run `downLoadBtcDayDate.ipynb` to download historical price data
2. Run `Candle_stick2.ipynb` to generate candlestick images
3. Run `Candle_CNN.ipynb` to train and evaluate the model

## References

This implementation is based on the research paper "Convolutional neural network for stock price prediction using transfer learning" (<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756702>)
