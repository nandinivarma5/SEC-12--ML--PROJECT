"""Shared constants and preprocessing used by training and the API."""
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Zero is not a valid medical value in these columns -> treated as missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Allowed input ranges used by the API for validation: (min, max)
LIMITS = {
    "Pregnancies": (0, 20), "Glucose": (0, 300), "BloodPressure": (0, 200),
    "SkinThickness": (0, 100), "Insulin": (0, 900), "BMI": (0, 70),
    "DiabetesPedigreeFunction": (0, 3), "Age": (1, 120),
}

MODEL_NAMES = {"logistic_regression": "Logistic Regression",
               "random_forest": "Random Forest"}


def clean_frame(df, medians):
    """Replace zeros in ZERO_AS_MISSING columns with training medians."""
    df = df.copy()
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, medians[col])
    return df
