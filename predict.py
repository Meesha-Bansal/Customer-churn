# utils/predict.py

import pandas as pd

# def make_single_prediction(model, input_data: dict):
#     """
#     Predict churn for a single user.
#     :param model: Trained sklearn/xgboost model
#     :param input_data: Dict of user input features
#     :return: Predicted label (0 or 1)
#     """
#     input_df = pd.DataFrame([input_data])
#     prediction = model.predict(input_df)[0]
#     return int(prediction)

def make_single_prediction(model, input_df):
    """
    input_df: a preprocessed DataFrame (1 row)
    """
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probability of churn (class 1)
    return int(prediction), proba

def make_batch_prediction(model, df: pd.DataFrame):
    """
    Predict churn for a batch of users in a CSV file.
    :param model: Trained model
    :param df: DataFrame from uploaded CSV
    :return: List of predictions
    """
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]  # churn probability for class 1
    return predictions.tolist(), probabilities.tolist()
