import pandas as pd
from sklearn.model_selection import train_test_split


file_path = '/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/data_processing/data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'

# Učitavanje podataka
df = pd.read_csv(file_path)

categorical_cols = ['GenHlth', 'Age', 'Education', 'Income']

df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

X = df_processed.drop('Diabetes_binary', axis=1)
y = df_processed['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

train_set = pd.concat([X_train, y_train], axis=1)
val_set = pd.concat([X_val, y_val], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

output_dir = '/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/data_processing/prepared_data/'
train_set.to_csv(f'{output_dir}train_set.csv', index=False)
val_set.to_csv(f'{output_dir}validation_set.csv', index=False)
test_set.to_csv(f'{output_dir}test_set.csv', index=False)

print("Podaci su obrađeni i sačuvani u folderu 'prepared_data'. Spremni za DVC verzionisanje.")