# Custom Implementation of K-Means, Bagging, and Boosting in Python

This project presents custom implementations of K-Means clustering, Bagging, and Boosting algorithms in Python. The aim of this project is to provide insights into how Bagging and Boosting algorithms perform on a custom dataset, particularly in comparison to a supervised learning task.

## Overview

The project focuses on a dataset containing the following features:

- Height
- Diameter
- Weight
- Hue

The goal is to predict the material based on these features using supervised learning techniques.

## Results

### Boosting Algorithm Analysis

#### Experiment 1

- Number of base models: 1, 10, 50, 100
- Accuracy: [0.6667, 0.6667, 0.5, 0.5]

Observations:
- Increasing the number of base models does not improve accuracy.
- Accuracy drops from 0.6667 to 0.5 when increasing the number of base models.
- Overfitting may be occurring, as the boosting model loses its ability to generalize to new data.

#### Experiment 2

- Number of base models: 1, 10, 25, 50
- Accuracy: [0.5, 0.3333, 0.3333, 0.3333]

Observations:
- Increasing the number of base models does not improve accuracy.
- Accuracy drops from 0.5 to 0.3333 when increasing the number of base models.
- Underfitting may be occurring, indicating that the model is too simple to capture the patterns in the data.

### K-Means Clustering Analysis

- k=3, accuracy=0.5088
- k=6, accuracy=0.6053
- k=9, accuracy=0.5526

Observations:
- The accuracy varies with different values of k.
- The highest accuracy is achieved with k=6.

## Conclusion

- Bagging and Boosting algorithms may not necessarily improve accuracy with an increase in the number of base models.
- Overfitting and underfitting are possible issues to consider when implementing boosting algorithms.
- K-Means clustering can achieve reasonable accuracy, with the optimal number of clusters determined by experimentation.

## Usage

1. Clone the repository.
2. Ensure Python 3.x is installed.
3. Run the Python scripts for K-Means, Bagging, and Boosting algorithms.
4. Analyze the results and adjust parameters as needed.
