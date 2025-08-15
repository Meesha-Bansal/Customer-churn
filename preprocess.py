import pandas as pd
import numpy as np

import re
import warnings
warnings.filterwarnings('ignore')

FEATURE_NAME_MAP = {
    "gender": ["gender", "sex"],
    "seniorcitizen": ["seniorcitizen", "senior_citizen", "senior"],
    "partner": ["partner", "spouse","married"],
    "dependents": ["dependents", "children"],
    "tenure": ["tenure", "tenureinmonths", "tenureinmonth", "tenureinyears", "tenureinyear"],
    "phoneservice": ["phoneservice", "phone_service"],
    "multiplelines": ["multiplelines", "multiple_lines"],
    "internetservice": ["internetservice", "internet_service"],
    "onlinesecurity": ["onlinesecurity", "security"],
    "onlinebackup": ["onlinebackup", "backup"],
    "deviceprotection": ["deviceprotection", "deviceprotectionplan"],
    "techsupport": ["techsupport", "premiumsupport"],
    "streamingtv": ["streamingtv", "tv"],
    "streamingmovies": ["streamingmovies", "movies"],
    "contract": ["contract"],
    "paperlessbilling": ["paperlessbilling", "paperless_billing"],
    "paymentmethod": ["paymentmethod", "payment_method"],
    "monthlycharges": ["monthlycharges", "monthly_charge"],
    "totalcharges": ["totalcharges", "total_charge"],
    "churn": ["churn", "churned", "churnstatus", "customerstatus"],
}


class ChurnPredictionModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.label_column = None
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []

    def _clean_column_names(self, df):
        df.columns = [re.sub(r'[^a-zA-Z0-9]', '', col.lower()) for col in df.columns]
        return df

    def _clean_churn_values(self, df):
        churn_map_pos = {'churn', 'churned', 'yes', '1', 'true', 'left', 'exited', 'cancelled'}
        churn_map_neg = {'no', '0', 'false', 'stay', 'stayed', 'joined', 'active'}

        df['churn'] = df['churn'].astype(str).str.lower().str.strip()
        df['churn'] = df['churn'].apply(lambda x: 1 if x in churn_map_pos else (0 if x in churn_map_neg else np.nan))
        df = df.dropna(subset=['churn']).copy()
        df['churn'] = df['churn'].astype(int)
        return df
    
    def standardize_features(self,df, feature_map):
        renamed_cols = {}
        original_cols = df.columns.tolist()

        cleaned_cols = [re.sub(r'[^a-zA-Z0-9]', '', col.lower()) for col in original_cols]

        for std_name, variants in feature_map.items():
            for variant in variants:
                variant_clean = re.sub(r'[^a-zA-Z0-9]', '', variant.lower())
                if variant_clean in cleaned_cols:
                    matched_idx = cleaned_cols.index(variant_clean)
                    original_col = original_cols[matched_idx]
                    renamed_cols[original_col] = std_name

                    # Special handling: convert "tenure in years" to months
                    if std_name == "tenure" and "year" in variant_clean:
                        df[original_col] = df[original_col].astype(float) * 12
                    break  # stop after the first match

        df = df.rename(columns=renamed_cols)
        return df

    def setup_preprocessing(self, df):
    # Binary Yes/No handling
        for col in df.select_dtypes(include=['object', 'bool']).columns:
            unique_vals = df[col].dropna().astype(str).str.lower().unique()
            if set(unique_vals).issubset({'yes', 'no'}):
                df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0})

        # Gender encoding (male=0, female=1)
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.lower().map({'male': 0, 'female': 1})

        # Contract encoding (0 = month-to-month, 1 = one year, 2 = two year)
        if 'contract' in df.columns:
            df['contract'] = df['contract'].astype(str).str.lower().apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x))
            contract_map = {
                'monthtomonth': 0,
                'oneyear': 1,
                'twoyear': 2
            }
            df['contract'] = df['contract'].map(contract_map)

            # Fallback if missing
            if df['contract'].isnull().all():
                df['contract'] = 0
            else:
                mode_val = df['contract'].mode()
                if not mode_val.empty:
                    df['contract'].fillna(mode_val[0], inplace=True)
                else:
                    df['contract'] = df['contract'].fillna(0)

        # 🔁 General fallback handling for string service features
        # Common "no service" patterns
        rejection_keywords = {
            'no', 'none', 'nointernet', 'nointernetservice', 'notapplicable', 'na', '0', 'null', 'unknown'
        }

        # Process all object/string columns (except contract/gender already handled)
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['gender', 'contract','totalcharges']:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].apply(
                    lambda x: 0 if x.replace(" ", "") in rejection_keywords else 1
                )

    def load_and_preprocess_data(self, df):
        # Drop rows with negative numeric values
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)

        # Then fill as before
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col].fillna(df[col].median(), inplace=True)


        # Sort by customerId if exists
        if 'customerid' in df.columns:
            df.sort_values(by='customerid', inplace=True)

         # 🔒 FIX HERE: Only clean churn if it's present
        if 'churn' in df.columns:
            self.label_column = df['churn']
            df = self._clean_churn_values(df)
        else:
            self.label_column = None

        # Derive seniorcitizen if age is available but not seniorcitizen
        if 'age' in df.columns and 'seniorcitizen' not in df.columns:
            df['seniorcitizen'] = (df['age'] >= 60).astype(int)
            df.drop(columns=['age'], inplace=True)

        # Step 1: Ensure columns exist and are binary
        df['phoneservice'] = df['phoneservice'] if 'phoneservice' in df.columns else 0
        df['internetservice'] = df['internetservice'] if 'internetservice' in df.columns else 0

        df['phoneservice'] = df['phoneservice'].astype(int)
        df['internetservice'] = df['internetservice'].astype(int)

        # Step 2: Initialize the new combined value to 0
        df['phoneservice_combined'] = 0

        # Step 3: Set value 2 where both are 1 (AND)
        df.loc[(df['phoneservice'] == 1) & (df['internetservice'] == 1), 'phoneservice_combined'] = 2

        # Step 4: Set value 1 where only one is 1 (OR) but not both
        df.loc[((df['phoneservice'] == 1) | (df['internetservice'] == 1)) &
            ~((df['phoneservice'] == 1) & (df['internetservice'] == 1)), 'phoneservice_combined'] = 1

        # Optional: replace original phoneservice column if desired
        df['phoneservice'] = df['phoneservice_combined']
        df.drop(columns=['phoneservice_combined'], inplace=True)


        # Feature engineering for online-related services
        online_cols = [col for col in ['onlinesecurity', 'onlinebackup', 'techsupport', 'deviceprotection'] if col in df.columns]
        if online_cols:
            df['onlineservice'] = df[online_cols].apply(
                lambda row: int(any(str(val).strip().lower() in ['yes', '1', 'true'] or val == 1 for val in row)), axis=1)
        else:
            df['onlineservice'] = 0

       # Feature engineering for streaming
        streaming_cols = [col for col in ['streamingtv', 'streamingmovies', 'streamingmusic'] if col in df.columns]
        if streaming_cols:
            df['streaming'] = df[streaming_cols].apply(
                lambda row: int(any(str(val).strip().lower() in ['yes', '1', 'true'] or val == 1 for val in row)), axis=1)
        else:
            df['streaming'] = 0

        # Final features (only use those available)
        final_features = ['gender', 'seniorcitizen', 'partner', 'tenure',
                        'phoneservice', 'onlineservice', 'streaming',
                        'contract', 'monthlycharges', 'totalcharges', 'churn']

        available_features = [col for col in final_features if col in df.columns]
        df = df[available_features].copy()

        # Clean numeric values
        if 'totalcharges' in df.columns:
            df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
            df['totalcharges'].fillna(df['totalcharges'].median(), inplace=True)
        # Fill missing numeric values with median
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col].fillna(df[col].median(), inplace=True)

        # Fill missing categorical values with mode
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
        # 
        return df
    
    

        

