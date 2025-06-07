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
miles_per_day = miles / days
receipts_per_day = receipts / days
log_miles = np.log1p(miles)
log_receipts = np.log1p(receipts)
days_squared = days ** 2
is_five_day_trip = int(days == 5)
is_sweet_spot_miles_per_day = int(180 <= miles_per_day <= 220)
low_receipts_penalty = int(receipts < 75 * days)
high_receipts_penalty = int(receipts > 120 * days)

features = np.array([[days, miles, receipts, miles_per_day, receipts_per_day,
                      log_miles, log_receipts, days_squared, is_five_day_trip,
                      is_sweet_spot_miles_per_day, low_receipts_penalty, high_receipts_penalty]])

# Predict and print
pred = model.predict(features)[0]
print(f"{pred:.2f}")
