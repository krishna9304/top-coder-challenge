import sys
import numpy as np
import joblib

# Load model
model = joblib.load('reimbursement_model_gbr_log.joblib')

# Parse inputs
days = float(sys.argv[1])
miles = float(sys.argv[2])
receipts = float(sys.argv[3])

# Feature engineering (must match training)
log_miles = np.log1p(miles)
log_receipts = np.log1p(receipts)

features = np.array([[days, miles, receipts, log_receipts, log_miles]])

# Predict and print
pred = model.predict(features)[0]
print(f"{pred:.2f}")
