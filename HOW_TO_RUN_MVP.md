# 🚀 How to Run Your Fraud Detection MVP

## Current Status
✅ Backend structure ready
✅ Frontend ready
✅ Basic models trained (Random Forest, XGBoost)
⚠️ **Deep learning models need to be trained** (LSTM, CNN)

---

## 📋 Step-by-Step Guide

### **Step 1: Install Dependencies First** ⏱️ (5-10 minutes)

Open **Command Prompt** or **PowerShell** and navigate to your project:

```bash
cd c:\xai-fincrime-poc-starter\xai-fincrime-poc
```

Install required packages (including TensorFlow for deep learning):

```bash
pip install -r requirements.txt
```

This will install TensorFlow, Keras, and other ML libraries. Wait for it to complete.

---

### **Step 2: Train the Deep Learning Models** ⏱️ (40-60 minutes)

Now navigate to the backend folder:

```bash
cd backend
```

Then run the training script:

```bash
python train_deep_learning_models.py
```

**What this does:**
- Trains LSTM model (for sequential patterns)
- Trains CNN model (for feature extraction)
- Creates ensemble model combining all 4 models
- Saves all models to `data/models/` folder

**Expected output:**
```
Loading data...
Training Random Forest...
Training XGBoost...
Creating sequences for LSTM/CNN...
Training LSTM model...
Epoch 1/50 - loss: 0.35 - accuracy: 0.87
...
Training CNN model...
Epoch 1/50 - loss: 0.33 - accuracy: 0.88
...
All models trained successfully!
```

---

### **Step 3: Start the Backend API** 🔧

Open a **NEW Command Prompt** window and run:

```bash
cd c:\xai-fincrime-poc-starter\xai-fincrime-poc\backend
uvicorn app.main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

✅ **Backend is now running!** Keep this window open.

---

### **Step 4: Start the Frontend** 💻

Open **ANOTHER Command Prompt** window and run:

```bash
cd c:\xai-fincrime-poc-starter\xai-fincrime-poc\frontend
npm install
npm start
```

**Expected output:**
```
  VITE v5.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

✅ **Frontend is now running!**

---

### **Step 5: Open the Application** 🌐

1. Open your web browser
2. Go to: **http://localhost:5173**
3. You should see the fraud detection interface!

---

## 🎯 How to Test Fraud Detection

### **Testing via the Web Interface**

Once the frontend loads, you'll see a form where you can enter transaction details:

1. **Enter a suspicious transaction** (likely fraud):
   - Step: `1`
   - Type: `TRANSFER`
   - Amount: `181.00`
   - Old Balance Origin: `181.00`
   - New Balance Origin: `0.00`
   - Old Balance Destination: `0.00`
   - New Balance Destination: `0.00`

2. Click **"Check Transaction"**

3. **You'll see results showing:**
   - Overall fraud probability (e.g., 73.4%)
   - Risk level (Low, Medium, High)
   - Individual predictions from all 4 models:
     - Random Forest: 68%
     - XGBoost: 75%
     - LSTM: 72%
     - CNN: 78%
   - Ensemble weights showing how each model contributes
   - **Explanations** showing why the transaction was flagged

### **Example Fraud Transaction (High Risk)**
```
Step: 1
Type: TRANSFER
Amount: 10000.00
Old Balance Origin: 10000.00
New Balance Origin: 0.00
Old Balance Destination: 0.00
New Balance Destination: 0.00
```

**Why it's flagged:**
- Origin account completely emptied
- Destination account was empty (suspicious)
- Large transfer amount
- TRANSFER type (high-risk)

### **Example Normal Transaction (Low Risk)**
```
Step: 5
Type: PAYMENT
Amount: 50.00
Old Balance Origin: 1000.00
New Balance Origin: 950.00
Old Balance Destination: 500.00
New Balance Destination: 550.00
```

**Why it's safe:**
- Small amount
- Normal balance changes
- PAYMENT type (lower risk)
- Both accounts have reasonable balances

---

## 🔍 How to See Explanations

The application provides two types of explanations:

### **1. For Risk Analysts** (Technical View)
Shows:
- SHAP values (feature importance)
- Model confidence scores
- Technical metrics (ROC-AUC, Precision, Recall)
- LSTM attention weights (which time steps matter)
- CNN Grad-CAM (which features matter)

### **2. For Compliance Officers** (Business View)
Shows:
- Risk level (High, Medium, Low)
- Key risk factors in plain language
- Recommended actions
- Regulatory notes
- Compliance thresholds

---

## 📊 Testing via API (Optional)

If you prefer to test directly via API without the frontend:

### **Test Prediction Endpoint**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"step\": 1, \"type\": \"TRANSFER\", \"amount\": 181.00, \"oldbalanceOrg\": 181.00, \"newbalanceOrig\": 0.00, \"oldbalanceDest\": 0.00, \"newbalanceDest\": 0.00}"
```

### **Test Explanation Endpoint (Risk Analyst)**
```bash
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d "{\"transaction\": {\"step\": 1, \"type\": \"TRANSFER\", \"amount\": 181.00, \"oldbalanceOrg\": 181.00, \"newbalanceOrig\": 0.00, \"oldbalanceDest\": 0.00, \"newbalanceDest\": 0.00}, \"stakeholder\": \"risk_analyst\"}"
```

### **Test Explanation Endpoint (Compliance Officer)**
```bash
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d "{\"transaction\": {\"step\": 1, \"type\": \"TRANSFER\", \"amount\": 181.00, \"oldbalanceOrg\": 181.00, \"newbalanceOrig\": 0.00, \"oldbalanceDest\": 0.00, \"newbalanceDest\": 0.00}, \"stakeholder\": \"compliance_officer\"}"
```

---

## ⚠️ Troubleshooting

### **Problem: "Module not found" errors**
**Solution:** Make sure you installed dependencies:
```bash
cd c:\xai-fincrime-poc-starter\xai-fincrime-poc
pip install -r requirements.txt
```

### **Problem: "Port 8000 already in use"**
**Solution:** Change the port:
```bash
uvicorn app.main:app --reload --port 8001
```
(Then update frontend API URL to http://localhost:8001)

### **Problem: Frontend shows connection error**
**Solution:**
1. Make sure backend is running on port 8000
2. Check if you see "Uvicorn running..." message
3. Test backend directly: http://localhost:8000/health

### **Problem: Models not loading**
**Solution:** You need to train the models first (Step 1)

### **Problem: Training takes too long**
**Solution:**
- This is normal! LSTM/CNN training takes 40-60 minutes on CPU
- If you have a GPU, it will be faster (12-23 minutes)
- You can reduce epochs in `train_deep_learning_models.py` (change from 50 to 20)

---

## 📱 What You'll See in the UI

1. **Input Form** - Enter transaction details
2. **Prediction Results** - Overall fraud probability and risk level
3. **Model Breakdown** - Individual predictions from all 4 models
4. **Ensemble Weights** - How much each model contributes
5. **Explanations** - Why the transaction was flagged
6. **Risk Factors** - Key indicators in plain language
7. **Recommended Actions** - What to do next

---

## 🎓 For Your FYP Demo

When demonstrating to your supervisor/examiner:

1. **Show different transaction types:**
   - High-risk fraud (empty account, large transfer)
   - Medium-risk (unusual but not definitely fraud)
   - Low-risk normal transactions

2. **Explain the 4 models:**
   - Random Forest & XGBoost (traditional ML)
   - LSTM (sequential patterns over time)
   - CNN (feature extraction)
   - Ensemble (combines all 4)

3. **Show explanations:**
   - Technical view for analysts
   - Business view for compliance officers

4. **Highlight the explainability:**
   - SHAP values showing which features matter
   - Attention weights showing which time steps matter
   - Plain language explanations for non-technical users

---

## ✅ Quick Checklist

- [x] Step 1: Install dependencies (run `pip install -r requirements.txt`)
- [x] Step 2: Train models (run `train_deep_learning_models.py`)
- [ ] Step 3: Start backend (run `uvicorn app.main:app --reload`)
- [ ] Step 4: Start frontend (run `npm start`)
- [ ] Step 5: Open browser (http://localhost:5173)
- [ ] Step 6: Test with example transactions
- [ ] Step 7: View explanations

---

## 📁 Important Files

- **Backend API:** `backend/app/main.py`
- **Training Script:** `backend/train_deep_learning_models.py`
- **Frontend UI:** `frontend/src/App.js`
- **Models Folder:** `data/models/` (generated after training)
- **Dataset:** `data/processed/paysim_sample.csv`

---

## 🎉 Summary

Once everything is running:
- **Backend:** http://localhost:8000
- **Frontend:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs

You'll be able to:
- ✅ Enter transaction details
- ✅ See predictions from 4 models
- ✅ View ensemble prediction
- ✅ Get detailed explanations
- ✅ See risk levels and recommended actions

**Good luck with your FYP! 🎓**
