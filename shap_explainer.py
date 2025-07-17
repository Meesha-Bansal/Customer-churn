# utils/shap_explainer.py

import shap
import pandas as pd


def get_shap_values(model, input_data):
    # Safe conversion
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    # Return top features and their impact (summary form)
    result = dict(zip(
        input_df.columns,
        shap_values.values[0].tolist()
    ))
    return result

