# train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# === Path Configuration ===
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent  # Go up one level from 'codes' to project root

DATA_PATH = PROJECT_ROOT / "data" / "emi_prediction_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

# Create models directory if it doesn't exist
MODEL_DIR.mkdir(exist_ok=True)

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Path: {DATA_PATH}")
print(f"📂 Model Directory: {MODEL_DIR}")

# Verify data file exists
if not DATA_PATH.exists():
    print(f"\n❌ ERROR: Dataset not found at {DATA_PATH}")
    print(f"Please ensure the file exists at this location.")
    exit(1)

print("\n📥 Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Loaded dataset with shape: {df.shape}")

# === Targets ===
target_class = "emi_eligibility"
target_reg = "max_monthly_emi"

# Handle string labels → encoded integers
label_encoder = LabelEncoder()
df[target_class] = label_encoder.fit_transform(df[target_class])
print(f"📊 Target classes: {list(label_encoder.classes_)}")

# === Column definitions ===
categorical_cols = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "emi_scenario"
]
numeric_cols = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "requested_amount", "requested_tenure"
]

print(f"📋 Categorical columns: {len(categorical_cols)}")
print(f"📋 Numeric columns: {len(numeric_cols)}")

# === Clean up data ===
print("\n🧹 Cleaning data...")
df[categorical_cols] = df[categorical_cols].fillna("missing")

# Convert numeric columns to float safely
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_cols] = df[numeric_cols].fillna(0.0)
print("✅ Data cleaning complete")

X = df[categorical_cols + numeric_cols]
y_class = df[target_class]
y_reg = df[target_reg]

print(f"\n📊 Dataset info:")
print(f"   Features shape: {X.shape}")
print(f"   Classification target shape: {y_class.shape}")
print(f"   Regression target shape: {y_reg.shape}")

# === Preprocessing ===
print("\n🔧 Setting up preprocessing pipelines...")
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ("cat", encoder, categorical_cols),
    ("num", scaler, numeric_cols)
])

# === Split ===
print("✂️ Splitting data (80-20 train-test split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# === Train models ===
print("\n🚀 Training RandomForest Classifier...")
print("   (This may take 5-10 minutes...)")
clf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=250, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1,
        verbose=1
    ))
])
clf_pipeline.fit(X_train, y_train)
train_score = clf_pipeline.score(X_train, y_train)
test_score = clf_pipeline.score(X_test, y_test)
print(f"✅ Classifier trained!")
print(f"   Train accuracy: {train_score:.4f}")
print(f"   Test accuracy: {test_score:.4f}")

print("\n🚀 Training RandomForest Regressor...")
print("   (This may take 5-10 minutes...)")
reg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=250, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1,
        verbose=1
    ))
])
reg_pipeline.fit(Xr_train, yr_train)
train_score_r = reg_pipeline.score(Xr_train, yr_train)
test_score_r = reg_pipeline.score(Xr_test, yr_test)
print(f"✅ Regressor trained!")
print(f"   Train R²: {train_score_r:.4f}")
print(f"   Test R²: {test_score_r:.4f}")

# === Save artifacts ===
print("\n💾 Saving models and preprocessors...")
fitted_encoder = clf_pipeline.named_steps["preprocess"].named_transformers_["cat"]
fitted_scaler = clf_pipeline.named_steps["preprocess"].named_transformers_["num"]

# Save all artifacts
joblib.dump(clf_pipeline, MODEL_DIR / "best_classifier.joblib")
joblib.dump(reg_pipeline, MODEL_DIR / "best_regressor.joblib")
joblib.dump(fitted_encoder, MODEL_DIR / "encoder.joblib")
joblib.dump(fitted_scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

print("\n✅ Training complete! Files saved in /models:")
for f in sorted(MODEL_DIR.glob("*.joblib")):
    size_mb = f.stat().st_size / 1e6
    print(f"   ✓ {f.name} ({size_mb:.2f} MB)")

print(f"\n🎉 Success! You can now run the Streamlit app:")
print(f"   cd {PROJECT_ROOT}")
print(f"   streamlit run codes/app.py")