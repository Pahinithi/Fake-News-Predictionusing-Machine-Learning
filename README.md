# Fake News Prediction using Machine Learning

This repository contains a machine learning project to predict the authenticity of news articles. The model is trained to distinguish between real and fake news using a logistic regression algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Web Application](#web-application)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The goal of this project is to build a model that can accurately predict whether a given news article is real or fake. The project uses natural language processing (NLP) techniques and a logistic regression model to achieve this goal.

## Dataset

The dataset used in this project is the Fake News dataset, which can be found in the `fake-news/train.csv` file. This dataset contains labeled news articles which are used for training and testing the model.

## Installation

To run this project, you'll need to have Python and the following libraries installed:

- Flask
- Numpy
- Scikit-learn
- NLTK
- Pickle

You can install the necessary libraries using the following command:

```bash
pip install Flask numpy scikit-learn nltk
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fake-news-prediction.git
    cd fake-news-prediction
    ```

2. Run the Flask app:
    ```bash
    python app.py
    ```

3. Open your browser and go to `http://127.0.0.1:5000/` to access the web application.

## Model

The machine learning model used in this project is a logistic regression classifier. The model is trained using the preprocessed text data from the news articles. The preprocessing steps include:

- Removing non-alphabetic characters
- Converting text to lowercase
- Tokenizing text
- Removing stopwords
- Stemming words

The trained model is saved as `fake_news_prediction_model.pkl`.

## Web Application

The web application allows users to input news content and get a prediction on whether the news is real or fake. The web app is built using Flask and consists of a simple HTML form where users can input the news content. The prediction result is displayed on the same page.

The main files for the web application are:
- `index.html`: The HTML template for the web page
- `app.py`: The Flask application script

## Results

The model is evaluated on a test set, and the performance metrics include accuracy, precision, recall, and F1-score. The detailed results and analysis can be found in the Jupyter notebook `fake-news-prediction-using-ml-logistic-regression - Jupyter Notebook.pdf`.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- The dataset used in this project is provided by [Kaggle](https://www.kaggle.com/).
