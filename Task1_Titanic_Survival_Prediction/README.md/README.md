Titanic Survival Prediction

1. Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning. It uses passenger information such as age, gender, passenger class, fare, and family details to make predictions.

2. Objective

The main objective is to build a machine learning classification model that predicts passenger survival based on historical Titanic passenger data.

3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

4. Dataset

The project uses the Titanic "train.csv" dataset. It contains passenger details and the target column "Survived".

- "0" = Did not survive
- "1" = Survived

5. Machine Learning Workflow

1. Load the dataset.
2. Select relevant input features.
3. Handle missing values.
4. Encode categorical variables.
5. Scale numerical features.
6. Split the dataset into training and testing sets.
7. Train a Logistic Regression model.
8. Evaluate model performance.
9. Save the trained model and evaluation results.

6. Model Evaluation

The model was evaluated using a test dataset.

- Accuracy: Approximately 80%
- Evaluation metrics: Precision, Recall, F1-score, and Support
- Model: Logistic Regression

7. Project Structure

Task1_Titanic_Survival_Prediction/
├── Data/
│   └── train.csv
├── Outputs/
│   ├── confusion_matrix.png
│   ├── evaluation.txt
│   └── titanic_model.joblib
├── main.py
├── requirements.txt
└── README.md

8. How to Run

Install the required libraries:

python -m pip install -r requirements.txt

Run the project:

python main.py

9. Output

The project generates:

- A trained machine learning model
- A confusion matrix visualization
- An evaluation report

10. Conclusion

This project demonstrates the basic machine learning workflow, including data preprocessing, model training, evaluation, and saving a trained model. It is a beginner-friendly classification project.