import pandas as pd
import numpy as np

# Step A: Load the 7 datasets
student_info = pd.read_csv('studentInfo.csv')
student_vle = pd.read_csv('studentVle.csv')
student_assessment = pd.read_csv('studentAssessment.csv')
student_registration = pd.read_csv('studentRegistration.csv')
# (Load courses.csv, assessments.csv, vle.csv similarly if needed)

print("Data loaded successfully!")

# Step B: Aggregate the Behavioral Data (studentVle)
# Calculate total engagement (sum_click) per student
student_clicks = student_vle.groupby('id_student')['sum_click'].sum().reset_index()

# Step C: Aggregate Assessment Performance (studentAssessment)
# Calculate the average assignment score for each student
student_scores = student_assessment.groupby('id_student')['score'].mean().reset_index()
student_scores.rename(columns={'score': 'average_score'}, inplace=True)

# Step D: Base Master Merge
# Combine demographics with registration records
master_df = pd.merge(student_info, student_registration, on=['id_student', 'code_module', 'code_presentation'], how='left')

# Merge the aggregated clicks (Behavioral Log)
master_df = pd.merge(master_df, student_clicks, on='id_student', how='left')

# Merge the aggregated performance scores
master_df = pd.merge(master_df, student_scores, on='id_student', how='left')

# Fill missing click or score records with 0
master_df['sum_click'] = master_df['sum_click'].fillna(0)
master_df['average_score'] = master_df['average_score'].fillna(0)

# Step E: Filter for your target target goal (Pass vs Withdrawn)
# As requested in IMG-20260701-WA0017.jpg to handle the 3:1 imbalance
filtered_df = master_df[master_df['final_result'].isin(['Pass', 'Withdrawn'])].copy()

print(f"Master dataset ready for machine learning! Shape: {filtered_df.shape}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Fill any missing values in numerical columns safely
filtered_df['date_registration'] = filtered_df['date_registration'].fillna(0)

# 2. Convert categorical columns to numeric using Label Encoding
label_encoders = {}
categorical_cols = ['code_module', 'code_presentation', 'gender', 'region', 
                    'highest_education', 'imd_band', 'age_band', 'disability']

for col in categorical_cols:
    le = LabelEncoder()
    filtered_df[col] = le.fit_transform(filtered_df[col].astype(str))
    label_encoders[col] = le

# 3. Target Variable: Map 'Pass' to 1 and 'Withdrawn' to 0
filtered_df['final_result'] = filtered_df['final_result'].map({'Pass': 1, 'Withdrawn': 0})

# Define Features (X) and Target (y)
X = filtered_df.drop(columns=['id_student', 'final_result', 'id_site'], errors='ignore')
y = filtered_df['final_result']

print("Features mapped. Sample shapes:", X.shape, y.shape)


from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

# 1. Isolate target variable
y = filtered_df['final_result']

# 2. Drop columns that are completely strings, IDs, or the target itself
# We drop text-heavy or high-cardinality IDs that shouldn't be scaled by SMOTE
X = filtered_df.drop(columns=['id_student', 'final_result', 'id_site', 'highest_education', 'imd_band', 'age_band'], errors='ignore')

# 3. Handle remaining categorical columns explicitly
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Encoding these text columns found in X:", categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 4. Fill any remaining NaN values with 0
X = X.fillna(0)

# Double Check Data Types before running SMOTE
print("\nVerify all columns are numbers:\n", X.dtypes)

# 5. Apply SMOTE safely
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nSMOTE Executed Successfully!")
print("Balanced dataset shape:", X_resampled.shape)


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Recreate a temporary dataframe for visualization from the resampled data
eda_df = pd.DataFrame(X_resampled, columns=X.columns)
eda_df['final_result'] = y_resampled

# 2. Map back labels for an elegant plot legend
eda_df['Status'] = eda_df['final_result'].map({1: 'Pass', 0: 'Withdrawn'})

# 3. Generate the Seaborn Pairplot directly (NO plt.figure() line here!)
# sns.pairplot creates its own canvas grid automatically.
g = sns.pairplot(eda_df, vars=['date_registration', 'average_score'], hue='Status', palette='Set1')

# 4. Correctly add the blueprint title to the automatic grid
g.fig.suptitle("EDA Blueprint: Registration Latency vs Assessment Performance", y=1.02)

# 5. Display the single, clean plot
plt.show()


from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import pandas as pd

# Calculate Information Gain (Mutual Information)
importances = mutual_info_classif(X_resampled, y_resampled, random_state=42)
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)

# Plot the Information Gain as requested in IMG-20260701-WA0017.jpg
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='teal')
plt.title("Information Gain: Behavioral Logs vs. Static Demographics")
plt.xlabel("Information Gain Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# Combine features and target back together to make a clean dashboard file
# final_export = X_resampled.copy()
# final_export['final_result'] = y_resampled

# Save it to your folder
# final_export.to_csv('oulad_cleaned_ml_data.csv', index=False)
# print("Data exported safely for Power BI!")
