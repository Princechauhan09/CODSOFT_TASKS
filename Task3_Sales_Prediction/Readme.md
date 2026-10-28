Sales Prediction Using Python

CodSoft Data Science Internship — Task 3

Project Overview

This project predicts product sales using advertising expenditure on TV, Radio, and Newspaper. It uses Machine Learning and a Linear Regression model to understand the relationship between advertising budgets and sales.

Objectives

- Analyze advertising and sales data.
- Train a Machine Learning regression model.
- Predict sales using advertising expenditure.
- Evaluate the model's performance using regression metrics.
- Visualize actual sales versus predicted sales.

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

Dataset

The project uses the Advertising dataset, which contains the following columns:

Column| Description
TV| Advertising expenditure on TV
Radio| Advertising expenditure on Radio
Newspaper| Advertising expenditure on Newspaper
Sales| Product sales

Machine Learning Algorithm

Linear Regression is used to predict sales based on TV, Radio, and Newspaper advertising expenditure.

The dataset is divided into training and testing sets using an 80:20 split.

Model Performance

The model achieved the following results in the test set:

Metric| Result
Mean Absolute Error (MAE)| 1.4608
Root Mean Squared Error (RMSE)| 1.7816
R² Score| 0.8994

The R² score indicates that the model explains approximately 89.94% of the variation in sales in this test evaluation.

Project Structure

Task3_Sales_Prediction/
├── Data/
│   └── Advertising.csv
├── Output/
│   ├── actual_vs_predicted.png
│   ├── evaluation.txt
│   ├── predictions.csv
│   └── sales_model.joblib
├── main.py
├── requirements.txt
└── README.md

Installation and Setup

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install the required libraries:

pip install -r requirements.txt

4. Run the project:

python main.py

Output Files

- "evaluation.txt" — model evaluation metrics.
- "predictions.csv" — actual and predicted sales values.
- "actual_vs_predicted.png" — visualization comparing actual and predicted sales.
- "sales_model.joblib" — saved trained model.

Conclusion

This project demonstrates how Linear Regression can be used for sales prediction based on advertising expenditure. It provides practical experience with data preprocessing, model training, evaluation, and visualization using Python.

Internship

CodSoft Data Science Internship — Task 3: Sales Prediction Using Python