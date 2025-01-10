# GitHub Repository Stars and Forks Predictor

This project is a deep learning-based tool to predict the **stars** and **forks** of GitHub repositories using various repository features. It uses the GitHub API to fetch repository data, preprocesses the data, and trains a neural network model to make predictions.

---

## Features
- Fetches repository data from GitHub API.
- Preprocesses data by filtering non-English descriptions and removing anomalies.
- Extracts key features such as:
  - Number of stars
  - Number of forks
  - Repository age
  - Description length
- Normalizes data and removes outliers using z-scores.
- Implements a deep neural network for prediction.
- Evaluates the model's performance using metrics like MSE, MAE, and R² score.
- Visualizes predictions and training progress.

---

## Requirements
To run this project, you need the following Python libraries:

- `requests`
- `numpy`
- `pandas`
- `re`
- `langdetect`
- `tensorflow`
- `scikit-learn`
- `matplotlib`

You can install these dependencies with:
```bash
pip install requests numpy pandas langdetect tensorflow scikit-learn matplotlib

Usage
Set Up Your GitHub Token: Replace your_personal_access_token in the code with your GitHub Personal Access Token.

Run the Script: Execute the script to fetch data, preprocess it, train the model, and visualize the results.

Results
Model Performance
Mean Squared Error (MSE):
Stars: X
Forks: X
Mean Absolute Error (MAE):
Stars: X
Forks: X
R² Score:
Stars: X
Forks: X
Visualizations
Loss and MAE over training epochs.
Predicted vs actual stars and forks.
