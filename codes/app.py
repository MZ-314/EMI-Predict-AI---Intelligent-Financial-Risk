import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ==========================================================
# 💰 EMIPredict AI – Intelligent Financial Risk Assessment
# ==========================================================
st.set_page_config(page_title="💰 EMIPredict AI", layout="centered")

st.title("💰 EMIPredict AI – Intelligent Financial Risk Assessment")
st.markdown("""
Predict EMI eligibility and maximum EMI limit using your financial data.  
_Powered by AI models trained on real credit datasets._
""")

# ==========================================================
# 🔧 Path Configuration
# ==========================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"

# ==========================================================
# 📦 Load Models and Preprocessors (Local Files)
# ==========================================================
@st.cache_resource
def load_models():
    """Load all models and preprocessors from local files."""
    try:
        st.info("📥 Loading models from local files...")
        
        clf = joblib.load(MODEL_DIR / "best_classifier.joblib")
        reg = joblib.load(MODEL_DIR / "best_regressor.joblib")
        encoder = joblib.load(MODEL_DIR / "encoder.joblib")
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
        
        st.success("✅ All models and preprocessors loaded successfully!")
        return clf, reg, encoder, scaler, label_encoder
    
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.error("Please run 'python train_model.py' first to generate the models.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Load all models
clf, reg, encoder, scaler, label_encoder = load_models()

# ==========================================================
# 🧮 User Input Section
# ==========================================================
st.header("🧮 Financial Information Input")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💵 Income & Expenses")
    salary = st.number_input("Monthly Salary (₹)", min_value=1000, value=50000, step=1000)
    current_emi = st.number_input("Current EMI (₹)", min_value=0, value=0, step=500)
    expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0, value=2000, step=500)
    travel = st.number_input("Travel Expenses (₹)", min_value=0, value=1000, step=500)
    groceries = st.number_input("Groceries & Utilities (₹)", min_value=0, value=5000, step=500)
    rent = st.number_input("Monthly Rent (₹)", min_value=0, value=8000, step=500)

with col2:
    st.subheader("🏦 Financial Status")
    loan = st.number_input("Existing Loans (₹)", min_value=0, value=0, step=500)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750, step=10)
    years = st.number_input("Years of Employment", min_value=0, value=5, step=1)
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=50000, value=500000, step=50000)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, max_value=84, value=60, step=6)

st.divider()

col3, col4, col5 = st.columns(3)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col4:
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
with col5:
    education = st.selectbox("Education", ["Graduate", "Postgraduate", "HighSchool"])

col6, col7, col8 = st.columns(3)
with col6:
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
with col7:
    company_type = st.selectbox("Company Type", ["Private", "Public", "Startup", "Other"])
with col8:
    house_type = st.selectbox("House Type", ["Owned", "Rented", "Company Provided"])

emi_scenario = st.selectbox("EMI Scenario", [
    "E-commerce Shopping EMI",
    "Home Appliances EMI", 
    "Vehicle EMI",
    "Personal Loan EMI",
    "Education EMI"
])

# ==========================================================
# 🔮 Prediction Section
# ==========================================================
st.divider()

if st.button("🔍 Predict EMI Eligibility & Maximum Limit", type="primary", use_container_width=True):
    try:
        # ✅ Prepare input DataFrame (matches training data format)
        df_input = pd.DataFrame([{
            "age": age,
            "monthly_salary": salary,
            "years_of_employment": years,
            "monthly_rent": rent,
            "family_size": 3,  # Default values (you can make these inputs too)
            "dependents": 1,
            "school_fees": 0,
            "college_fees": 0,
            "travel_expenses": travel,
            "groceries_utilities": groceries,
            "other_monthly_expenses": expenses,
            "existing_loans": loan,
            "current_emi_amount": current_emi,
            "credit_score": credit_score,
            "bank_balance": 50000,  # Default
            "emergency_fund": 10000,  # Default
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "employment_type": employment_type,
            "company_type": company_type,
            "house_type": house_type,
            "emi_scenario": emi_scenario
        }])

        # Show input summary
        with st.expander("📋 Review Your Input Data"):
            st.dataframe(df_input, use_container_width=True)

        # === Predict EMI Eligibility (Classification) ===
        class_pred_encoded = clf.predict(df_input)[0]
        class_label = label_encoder.inverse_transform([class_pred_encoded])[0]
        
        # Get prediction probabilities if available
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(df_input)[0]
            confidence = float(proba[class_pred_encoded])
        else:
            confidence = None

        # === Predict Maximum EMI Amount (Regression) ===
        max_emi = float(reg.predict(df_input)[0])

        # ==========================================================
        # 📊 Display Results
        # ==========================================================
        st.success("### ✅ Prediction Complete!")
        
        # Display eligibility result with color coding
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### 🏦 EMI Eligibility Status")
            if class_label == "Eligible":
                st.success(f"### ✅ {class_label}")
                st.markdown("**You are eligible for the EMI!**")
            elif class_label == "High_Risk":
                st.warning(f"### ⚠️ {class_label}")
                st.markdown("**Approval possible with higher interest rates**")
            else:  # Not_Eligible
                st.error(f"### ❌ {class_label}")
                st.markdown("**EMI approval not recommended**")
            
            if confidence:
                st.metric("Confidence", f"{confidence*100:.1f}%")
                st.progress(confidence)
        
        with col_b:
            st.markdown("#### 💸 Maximum Affordable EMI")
            st.metric(
                label="Recommended EMI Limit",
                value=f"₹{max_emi:,.0f}",
                delta=f"{(max_emi/salary)*100:.1f}% of salary"
            )
            
            # Calculate affordability ratio
            affordability_ratio = (current_emi + max_emi) / salary * 100
            st.info(f"**Total EMI Burden:** {affordability_ratio:.1f}% of monthly income")

        # ==========================================================
        # 📈 Financial Analysis
        # ==========================================================
        st.divider()
        st.markdown("### 📈 Financial Analysis")
        
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            debt_to_income = ((current_emi + loan) / salary * 100) if salary > 0 else 0
            st.metric("Debt-to-Income Ratio", f"{debt_to_income:.1f}%")
        
        with col_y:
            disposable_income = salary - (current_emi + rent + expenses + travel + groceries)
            st.metric("Disposable Income", f"₹{disposable_income:,.0f}")
        
        with col_z:
            expense_ratio = ((rent + expenses + travel + groceries) / salary * 100) if salary > 0 else 0
            st.metric("Expense Ratio", f"{expense_ratio:.1f}%")

        # Recommendations
        st.divider()
        st.markdown("### 💡 Recommendations")
        
        if class_label == "Eligible" and max_emi > 0:
            st.info("""
            ✅ **Great news!** Your financial profile is strong.
            - Your credit score is good
            - You have sufficient income to support the EMI
            - Maintain your current financial discipline
            """)
        elif class_label == "High_Risk":
            st.warning("""
            ⚠️ **Proceed with caution:**
            - Consider reducing the loan amount
            - Try to improve your credit score
            - Reduce existing debts before applying
            - You may face higher interest rates
            """)
        else:
            st.error("""
            ❌ **Not recommended at this time:**
            - Your current financial obligations are high
            - Consider waiting to improve your financial position
            - Focus on reducing existing debts
            - Work on improving your credit score
            """)

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)
        import traceback
        st.code(traceback.format_exc())

# ==========================================================
# 🧠 Model Information (Footer)
# ==========================================================
st.divider()
with st.expander("ℹ️ About the Models"):
    st.markdown("""
    ### Model Information
    
    **Classification Model:** Random Forest Classifier
    - Predicts: Eligible / High_Risk / Not_Eligible
    - Trained on: 323,840 samples
    - Test Accuracy: 91.05%
    
    **Regression Model:** Random Forest Regressor  
    - Predicts: Maximum affordable EMI amount
    - Trained on: 323,840 samples
    - Test R² Score: 95.92%
    
    **Dataset:** 400,000+ financial records across 5 EMI scenarios
    
    **Features:** 25 input variables including demographics, income, expenses, and credit history
    """)
    
    # Show feature information if available
    if hasattr(clf, "feature_names_in_"):
        with st.expander("🔍 Model Features"):
            st.write("**Features used by the model:**")
            st.write(list(clf.feature_names_in_))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform</p>
    <p><small>Powered by Machine Learning | Built with Streamlit</small></p>
</div>
""", unsafe_allow_html=True)