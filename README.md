# 💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

An intelligent ML-powered platform for predicting EMI eligibility and maximum affordable EMI amounts using financial and demographic data.

## 🎯 Features

- **Dual ML Problem Solving**
  - Classification: EMI Eligibility (Eligible/High_Risk/Not_Eligible)
  - Regression: Maximum Affordable EMI Amount

- **High Accuracy**
  - Classification: 91.05% test accuracy
  - Regression: 95.92% R² score

- **Comprehensive Financial Analysis**
  - Real-time risk assessment
  - Debt-to-income ratio calculation
  - Personalized recommendations

- **User-Friendly Interface**
  - Interactive Streamlit web application
  - Instant predictions
  - Visual financial analysis dashboard

## 📊 Dataset

- **Records:** 400,000+ financial profiles
- **Features:** 25 input variables
- **EMI Scenarios:** 5 lending categories
  - E-commerce Shopping EMI
  - Home Appliances EMI
  - Vehicle EMI
  - Personal Loan EMI
  - Education EMI

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "09_EMIPredict AI - Intelligent Financial Risk"
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### ⚠️ Important: Model Files Setup

**The trained model files are NOT included in this repository** due to GitHub's file size limits (models are ~1GB total).

#### Option 1: Train Models Locally (Recommended)

If you have the dataset file:

```bash
cd codes
python train_model.py
```

This will:
- Train the Random Forest models
- Save them to the `models/` directory
- Take approximately 10-15 minutes

#### Option 2: Download Pre-trained Models

Download the pre-trained models from [Google Drive / OneDrive / etc.] and place them in the `models/` directory:

Required files:
- `best_classifier.joblib` (~383 MB)
- `best_regressor.joblib` (~665 MB)
- `encoder.joblib`
- `scaler.joblib`
- `label_encoder.joblib`

### Running the Application

```bash
streamlit run codes/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
09_EMIPredict AI - Intelligent Financial Risk/
├── codes/
│   ├── app.py                      # Streamlit web application
│   └── train_model.py              # Model training script
├── data/
│   └── emi_prediction_dataset.csv  # Training dataset (not in repo)
├── models/                         # Model files (not in repo)
│   ├── best_classifier.joblib      # Random Forest Classifier
│   ├── best_regressor.joblib       # Random Forest Regressor
│   ├── encoder.joblib              # Categorical encoder
│   ├── scaler.joblib               # Numerical scaler
│   └── label_encoder.joblib        # Label encoder
├── docs/
│   └── EMIPredict AI.pdf           # Project documentation
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **ML Framework:** scikit-learn
- **Models:** Random Forest (Classifier & Regressor)
- **Data Processing:** pandas, numpy
- **Model Persistence:** joblib

## 📈 Model Performance

### Classification Model (EMI Eligibility)
- **Algorithm:** Random Forest Classifier
- **Training Accuracy:** 93.89%
- **Test Accuracy:** 91.05%
- **Trees:** 250
- **Max Depth:** 15

### Regression Model (Maximum EMI)
- **Algorithm:** Random Forest Regressor
- **Training R²:** 98.26%
- **Test R²:** 95.92%
- **Trees:** 250
- **Max Depth:** 15

## 🎓 Model Features (25 Variables)

### Demographics
- Age, Gender, Marital Status, Education

### Employment
- Monthly Salary, Employment Type, Years of Employment, Company Type

### Housing
- House Type, Monthly Rent, Family Size, Dependents

### Expenses
- School Fees, College Fees, Travel Expenses
- Groceries & Utilities, Other Monthly Expenses

### Financial Status
- Existing Loans, Current EMI Amount, Credit Score
- Bank Balance, Emergency Fund

### Loan Request
- Requested Amount, Requested Tenure, EMI Scenario

## 🌐 Deployment Options

### Streamlit Cloud
1. Push code to GitHub (models will need to be downloaded/trained)
2. Connect repository to Streamlit Cloud
3. Set main file: `codes/app.py`
4. Deploy!

### Local Deployment
- Already set up! Just run `streamlit run codes/app.py`

## 🔒 Security & Privacy

- No data is stored or transmitted externally
- All predictions happen locally
- Models run entirely on your machine/server

## 📝 Usage Example

1. Enter your financial information
2. Provide demographic details
3. Specify loan requirements
4. Click "Predict"
5. View your eligibility status and maximum EMI
6. Review personalized recommendations

## 🤝 Contributing

This is an educational/capstone project. For questions or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is for educational purposes as part of the EMIPredict AI capstone project.

## 👨‍💻 Author

[Your Name]

## 🙏 Acknowledgments

- Dataset: EMI Prediction Dataset (400K records)
- Framework: Streamlit for the amazing UI framework
- ML Library: scikit-learn for robust ML algorithms

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review project documentation in `docs/`
- Ensure all dependencies are installed correctly

---

**Note:** The dataset and trained models are not included in this repository due to size constraints. Please follow the setup instructions above to obtain them.