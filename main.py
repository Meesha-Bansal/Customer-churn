from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback

from utils.predict import make_single_prediction, make_batch_prediction
from utils.shap_explainer import get_shap_values
from utils.report_generator import generate_pdf_report
from utils.preprocess import ChurnPredictionModel, FEATURE_NAME_MAP

app = FastAPI(title="Customer Churn Prediction API")

# Load model once at startup
trained_model = joblib.load("model/best_churn_model.pkl")


# ===== Input Schema for Single User =====
class UserInput(BaseModel):
    gender: int
    seniorcitizen: int
    partner: int
    tenure: float
    phoneservice: int
    onlineservice: int
    streaming: int
    contract: int
    monthlycharges: float
    totalcharges: float


# ======= ROUTES =======

@app.post("/predict")
async def predict_single(data: UserInput):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])

        # Run through preprocessing pipeline
        processor = ChurnPredictionModel()
        df = processor._clean_column_names(df)
        df = processor.standardize_features(df, FEATURE_NAME_MAP)
        processor.setup_preprocessing(df)
        df = processor.load_and_preprocess_data(df)

        # Drop churn column if accidentally added
        if 'churn' in df.columns:
            df.drop(columns=['churn'], inplace=True)

        # Predict + SHAP
        prediction, probability = make_single_prediction(trained_model, df)
        explanation = get_shap_values(trained_model, df)

        return {
            "prediction": int(prediction),
            "probability": float(round(probability, 4)),
            "shap": {str(k): float(v) for k, v in explanation.items()}
        }


    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Preprocess
        processor = ChurnPredictionModel()
        df = processor._clean_column_names(df)
        df = processor.standardize_features(df, FEATURE_NAME_MAP)
        processor.setup_preprocessing(df)
        df = processor.load_and_preprocess_data(df)

        # Separate features
        if 'churn' in df.columns:
            df = df.drop(columns=['churn'])

        X = df.copy()
        predictions, probabilities = make_batch_prediction(trained_model, X)

        # Compute SHAP values
        import shap
        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(X)

        # Convert SHAP values into a list of dictionaries for each row
        shap_dict_list = pd.DataFrame(shap_values, columns=X.columns).to_dict(orient="records")

        # return {"predictions": predictions, "shap": shap_dict_list}
        result_data = [
            {
                "prediction": int(pred),
                "probability": float(round(prob, 4)),
                "shap": {str(k): float(v) for k, v in shap_dict.items()}
            }
            for pred, prob, shap_dict in zip(predictions, probabilities, shap_dict_list)
        ]


        return {"results": result_data}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )




