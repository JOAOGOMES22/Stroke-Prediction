# A list of columns selected for analysis or model input.
# These columns are expected to be present in the dataset and represent 
# key factors potentially influencing stroke risk or related health outcomes.
SELECTED_COLUMNS = [
    "Age",                     # Age of the individual (numeric)
    "Gender",                  # Gender of the individual (categorical)
    "Hypertension",            # Presence of hypertension (binary: 0=No, 1=Yes)
    "Average Glucose Level",   # Average blood glucose level (numeric)
    "Smoking Status",          # Smoking behavior/status (categorical)
    "Heart Disease",           # Presence of heart disease (binary: 0=No, 1=Yes)
    "Alcohol Intake",          # Alcohol consumption level (numeric or categorical)
    "Physical Activity",       # Level of physical activity (categorical or numeric)
    "Stress Levels",           # Stress levels on a defined scale (numeric)
    "Family History of Stroke",# Binary indicator for family history of stroke (0=No, 1=Yes)
    "Dietary Habits",          # Dietary behavior or pattern (categorical or descriptive)
]
