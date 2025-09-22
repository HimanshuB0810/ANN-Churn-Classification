# Customer Churn and Salary Prediction ANN

This project contains two machine learning models built with Artificial Neural Networks (ANNs) to predict customer churn and estimate customer salary from the same dataset. Both models are deployed as interactive web applications using Streamlit.

-----

## 📜 Project Overview

This repository showcases the implementation of two distinct ANN models for:

1.  **Churn Classification**: A classification model that predicts whether a customer is likely to churn (leave the bank) or not. The corresponding notebook for this model is `experiments.ipynb` and the Streamlit application is `app.py`.
2.  **Salary Regression**: A regression model that predicts the estimated salary of a customer. The notebook for this model is `salary_regression.ipynb` and the Streamlit application is `streamlit_regression.py`.

-----

## ✨ Features

  * **Dual Models**: Implements both classification and regression models from a single dataset.
  * **Interactive UI**: User-friendly web interface built with Streamlit for both models to input customer data and get predictions.
  * **ANN Implementation**: Utilizes TensorFlow and Keras to build, train, and evaluate the neural network models.
  * **Data Preprocessing**: Demonstrates standard data preprocessing techniques like one-hot encoding for categorical variables and feature scaling.
  * **Saved Models and Encoders**: The trained models and encoders are saved as `.h5` and `.pkl` files respectively, for easy deployment and use in the Streamlit apps.

-----

## 💾 Dataset

The project uses the "Churn\_Modelling.csv" dataset which contains information about bank customers. The dataset includes features like credit score, geography, gender, age, tenure, balance, number of products, and whether the customer has a credit card.

-----

## ⚙️ Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/himanshub0810/ann-churn-classification.git
    cd ann-churn-classification
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## 🚀 Usage

You can run either of the two Streamlit applications:

  * **For Churn Prediction:**

    ```bash
    streamlit run app.py
    ```

  * **For Estimated Salary Prediction:**

    ```bash
    streamlit run streamlit_regression.py
    ```

After running the command, a new tab will open in your browser with the web application. You can then input the customer details in the sidebar to get the prediction.

-----

## 🧠 Model Details

Both models are built using a Sequential ANN architecture with the following layers:

  * **Input Layer**: Takes the preprocessed customer data as input.
  * **Hidden Layers**: Two hidden layers with ReLU activation function (64 and 32 neurons respectively).
  * **Output Layer**:
      * For the **classification model**, a single neuron with a sigmoid activation function is used to output the churn probability.
      * For the **regression model**, a single neuron with a linear activation function is used to output the estimated salary.

-----

## 💻 Technologies Used

  * **TensorFlow & Keras**: For building and training the ANN models.
  * **Streamlit**: For creating and deploying the interactive web applications.
  * **Pandas**: For data manipulation and analysis.
  * **Scikit-learn**: For data preprocessing (StandardScaler, OneHotEncoder, LabelEncoder).
  * **NumPy**: For numerical operations.
  * **Pickle**: For saving and loading the encoders and scaler.
  * **Jupyter Notebook**: For model development and experimentation.