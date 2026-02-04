# 💰 EMIPredict AI - Setup Guide

## 📋 Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

## 🚀 Quick Start Guide

### Step 1: Set Up Python Environment

```bash
# Navigate to project directory
cd "09_EMIPredict AI - Intelligent Financial Risk"

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Models (First Time Setup)

**Important:** You need to train the models before running the app, as the large model files (`best_classifier.joblib` and `best_regressor.joblib`) are downloaded from Google Drive in the app.

```bash
# Make sure you're in the project root directory
cd codes
python train_model.py
```

This will:
- Load the dataset from `data/emi_prediction_dataset.csv`
- Train the RandomForest classifier and regressor
- Save all models to the `models/` directory
- Generate the following files:
  - `best_classifier.joblib` (~100-200 MB)
  - `best_regressor.joblib` (~100-200 MB)
  - `encoder.joblib`
  - `scaler.joblib`
  - `label_encoder.joblib`

**⏱️ Training Time:** 5-15 minutes depending on your machine specs

### Step 4: Upload Models to Google Drive (For Streamlit Cloud)

Since the trained models are too large for GitHub:

1. Upload `best_classifier.joblib` and `best_regressor.joblib` to your Google Drive
2. Make them publicly accessible (Anyone with the link can view)
3. Get the file IDs from the share links:
   - Share link format: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - Extract the `FILE_ID` part
4. Update the file IDs in `app.py`:
   ```python
   clf_id = "YOUR_CLASSIFIER_FILE_ID"
   reg_id = "YOUR_REGRESSOR_FILE_ID"
   ```

### Step 5: Run the Streamlit App Locally

```bash
# From project root
streamlit run codes/app.py

# Or if you're already in the codes directory
streamlit run app.py
```

The app should open automatically in your browser at `http://localhost:8501`

## 📁 Project Structure

```
09_EMIPredict AI - Intelligent Financial Risk/
├── codes/
│   ├── app.py                      # Main Streamlit application
│   └── train_model.py              # Model training script
├── data/
│   └── emi_prediction_dataset.csv  # Training dataset (400K records)
├── models/                         # Generated after training
│   ├── best_classifier.joblib      # ~150 MB (from Google Drive)
│   ├── best_regressor.joblib       # ~150 MB (from Google Drive)
│   ├── encoder.joblib              # Categorical encoder
│   ├── scaler.joblib               # Numerical scaler
│   └── label_encoder.joblib        # Target label encoder
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## ⚠️ Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Issue: "FileNotFoundError: data/emi_prediction_dataset.csv"
```bash
# Solution: Ensure you're running from the correct directory
# train_model.py should be run from the 'codes' directory
# OR update DATA_PATH in train_model.py to use absolute path
```

### Issue: Models not downloading from Google Drive
```bash
# Solution: Check if file IDs are correct and files are publicly accessible
# Test the download manually:
# https://drive.google.com/uc?id=YOUR_FILE_ID
```

### Issue: "MemoryError during training"
```python
# Solution: Reduce model complexity in train_model.py
# Change n_estimators from 250 to 100:
RandomForestClassifier(n_estimators=100, max_depth=10, ...)
```

## 🌐 Deploying to Streamlit Cloud

1. Push your code to GitHub (exclude large model files)
2. Add model files to Google Drive as described above
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repository
5. Set main file path: `codes/app.py`
6. Deploy!

## 📊 Dataset Information

- **Records:** 400,000 financial profiles
- **Features:** 22 input variables
- **Targets:** 
  - Classification: EMI eligibility (Eligible/High_Risk/Not_Eligible)
  - Regression: Maximum monthly EMI amount

## 🛠️ Development Workflow

1. **Data Exploration:** Analyze the dataset in `data/`
2. **Feature Engineering:** Modify `train_model.py` for new features
3. **Model Training:** Run `python train_model.py`
4. **Testing:** Test predictions locally with `streamlit run app.py`
5. **Deployment:** Push to GitHub and deploy on Streamlit Cloud

## 📝 Notes

- The first run will download large models (~300 MB total) from Google Drive
- Models are cached after first download
- Training from scratch takes 5-15 minutes
- Ensure you have at least 2 GB free RAM for training
- The app uses sklearn pipelines for automatic preprocessing

## 🤝 Support

For issues or questions:
- Check the troubleshooting section above
- Review the project documentation in `docs/`
- Ensure all dependencies are correctly installed

## 📄 License

This project is for educational purposes as part of the EMIPredict AI capstone project.