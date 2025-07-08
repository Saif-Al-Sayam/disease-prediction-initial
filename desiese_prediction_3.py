import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_disease_data(disease_name):
    data_files = {
        "diabetes": "diabetes.csv",
        "heart": "heart_disease.csv",
        "liver": "liver_disease.csv"
    }

    if disease_name not in data_files:
        raise ValueError(f"Unknown disease: {disease_name}")

    file_path = data_files[disease_name]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}.")

    df = pd.read_csv(file_path)

    if disease_name == "heart":
        df = df.replace('?', np.nan)
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'target' in df.columns:
            df['target'] = (df['target'] > 0).astype(int)
    elif disease_name == "liver":
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        if 'Dataset' in df.columns:
            df = df.rename(columns={'Dataset': 'target'})

    return df


def train_disease_model(disease_name):
    try:
        df = load_disease_data(disease_name)

        disease_config = {
            "diabetes": {
                "target": 'Outcome',
                "features": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                "model": LogisticRegression(max_iter=1000)
            },
            "heart": {
                "target": 'target',
                "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                             'ca', 'thal'],
                "model": RandomForestClassifier(n_estimators=100, random_state=42)
            },
            "liver": {
                "target": 'target',
                "features": ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                             'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                             'Albumin_and_Globulin_Ratio'],
                "model": LogisticRegression(max_iter=1000)
            }
        }

        config = disease_config[disease_name]
        X = df[config["features"]]
        y = df[config["target"]]

        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        model = config["model"]
        model.fit(X_train, y_train)

        joblib.dump(model, f'{disease_name}_model.pkl')
        joblib.dump(scaler, f'{disease_name}_scaler.pkl')
        joblib.dump(imputer, f'{disease_name}_imputer.pkl')

        y_pred = model.predict(X_test)
        print(f"\n{disease_name} Model trained successfully!")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return model

    except Exception as e:
        print(f"\nError training {disease_name} model: {str(e)}")
        return None


def predict_disease():
    while True:
        print("\n--- Disease Prediction System ---")
        print("1. Diabetes")
        print("2. Heart Disease")
        print("3. Liver Disease")
        print("4. Exit")

        choice = input("Select option (1-4): ").strip()
        if choice == "4":
            print("\nExiting system. Goodbye!")
            break

        disease_map = {"1": "diabetes", "2": "heart", "3": "liver"}
        if choice not in disease_map:
            print("Invalid choice. Please select 1-4.")
            continue

        disease = disease_map[choice]

        if not all(os.path.exists(f'{disease}_{f}.pkl') for f in ['model', 'scaler', 'imputer']):
            print(f"\n{disease.capitalize()} model not found. Training new model...")
            if not train_disease_model(disease):
                print(f"Failed to train {disease} model. Please check your dataset.")
                continue

        try:
            model = joblib.load(f'{disease}_model.pkl')
            scaler = joblib.load(f'{disease}_scaler.pkl')
            imputer = joblib.load(f'{disease}_imputer.pkl')

            feature_descriptions = {
                "diabetes": [
                    ('Pregnancies', 'Number of pregnancies (0-20)'),
                    ('Glucose', 'Glucose level (mg/dL)'),
                    ('BloodPressure', 'Blood pressure (mmHg)'),
                    ('SkinThickness', 'Skin thickness (mm)'),
                    ('Insulin', 'Insulin level (μU/mL)'),
                    ('BMI', 'Body Mass Index (kg/m²)'),
                    ('DiabetesPedigreeFunction', 'Diabetes pedigree function (0.08-2.42)'),
                    ('Age', 'Age (years)')
                ],
                "heart": [
                    ('age', 'Age (years)'),
                    ('sex', 'Sex (1 = male, 0 = female)'),
                    ('cp', 'Chest pain type (1-4)'),
                    ('trestbps', 'Resting blood pressure (mmHg)'),
                    ('chol', 'Serum cholesterol (mg/dL)'),
                    ('fbs', 'Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)'),
                    ('restecg', 'Resting electrocardiographic results (0-2)'),
                    ('thalach', 'Maximum heart rate achieved'),
                    ('exang', 'Exercise induced angina (1 = yes, 0 = no)'),
                    ('oldpeak', 'ST depression induced by exercise relative to rest'),
                    ('slope', 'Slope of the peak exercise ST segment (1-3)'),
                    ('ca', 'Number of major vessels colored by fluoroscopy (0-3)'),
                    ('thal', 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)')
                ],
                "liver": [
                    ('Age', 'Age (years)'),
                    ('Gender', 'Gender (1 = male, 0 = female)'),
                    ('Total_Bilirubin', 'Total Bilirubin (mg/dL)'),
                    ('Direct_Bilirubin', 'Direct Bilirubin (mg/dL)'),
                    ('Alkaline_Phosphotase', 'Alkaline Phosphotase (IU/L)'),
                    ('Alamine_Aminotransferase', 'Alamine Aminotransferase (IU/L)'),
                    ('Aspartate_Aminotransferase', 'Aspartate Aminotransferase (IU/L)'),
                    ('Total_Protiens', 'Total Proteins (g/dL)'),
                    ('Albumin', 'Albumin (g/dL)'),
                    ('Albumin_and_Globulin_Ratio', 'Albumin and Globulin Ratio')
                ]
            }

            print(f"\nEnter the following information for {disease} prediction:")
            inputs = []
            for feature, description in feature_descriptions[disease]:
                while True:
                    try:
                        value = input(f"{description}: ")
                        if feature == 'Gender' and disease == 'liver':
                            value = 1 if value.lower() in ['male', 'm', '1'] else 0
                        else:
                            value = float(value)
                        inputs.append(value)
                        break
                    except ValueError:
                        print(f"Invalid input. Please enter a valid number for {feature}.")

            input_data = np.array(inputs).reshape(1, -1)
            input_data = imputer.transform(input_data)
            input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1] if hasattr(model, 'predict_proba') else None

            if disease == "liver":
                result = prediction == 1
            else:
                result = prediction == 1

            print("\n" + "=" * 40)
            print("Disease detected!" if result else "No disease detected")
            if proba:
                print(f"Confidence: {proba * 100:.1f}%")
            print("=" * 40)

        except Exception as e:
            print(f"\nError during prediction: {str(e)}")


if __name__ == "__main__":
    for disease in ['diabetes', 'heart', 'liver']:
        model_file = f'{disease}_model.pkl'
        if not os.path.exists(model_file):
            print(f"\nTraining {disease} model...")
            train_disease_model(disease)

    predict_disease()