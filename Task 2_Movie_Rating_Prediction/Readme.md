# Movie Rating Prediction with Python

## Project Overview
This project uses machine learning to predict movie ratings based on movie details from the IMDb Movies India dataset.

## Objective
To build a regression model that estimates movie ratings using features such as genre, director, actors, year, duration, and votes.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib

## Machine Learning Model
Random Forest Regressor

## Dataset
IMDb Movies India dataset. The model used 7,919 records after data cleaning.

## Evaluation Results
- Mean Absolute Error (MAE): 0.804
- Root Mean Squared Error (RMSE): 1.081
- R-squared (R2): 0.371

These results are baseline metrics; model performance may be improved with further feature engineering and tuning.

## Project Outputs
- `evaluation.txt` — evaluation metrics
- `actual_vs_predicted.png` — prediction comparison plot
- `movie_rating_model.joblib` — trained model
- `predictions.csv` — actual and predicted ratings

## How to Run
1. Install Python.
2. Install dependencies using `python -m pip install -r requirements.txt`.
3. Run the project using `python .\main.py\main.py` if the script is inside the nested folder, or `python .\main.py` if it is in the root folder.

## Conclusion
This project demonstrates a machine learning regression workflow for movie rating prediction.