# Property Price Prediction Model

## Model Information
- **Model Type**: Linear Regression (Degree 2)
- **Test R² Score**: 0.9476
- **Test RMSE**: 209.68 Juta Rp
- **Training Date**: 2025-10-29 19:55:55

## How to Use

```python
import joblib
import numpy as np
import pandas as pd

# Load model and transformers
model = joblib.load('best_property_price_model.pkl')
poly_transformer = joblib.load('best_poly_transformer.pkl')
scaler = joblib.load('property_scaler.pkl')

# Prepare your data
data = {
    'Luas_Tanah': 200.0,
    'Luas_Bangunan': 150.0,
    'Jarak_ke_Pusat_Kota': 10.0,
    'Jumlah_Kamar_Tidur': 3,
    'Umur_Bangunan': 5
}

# Scale continuous features
cont_features = scaler.transform([[data['Luas_Tanah'], data['Luas_Bangunan'], data['Jarak_ke_Pusat_Kota']]])

# Combine with integer features
features_scaled = np.column_stack([cont_features, [[data['Jumlah_Kamar_Tidur'], data['Umur_Bangunan']]]])

# Transform to polynomial
features_poly = poly_transformer.transform(features_scaled)

# Predict
predicted_price = model.predict(features_poly)[0]
print(f"Predicted Price: Rp {predicted_price:.2f} Juta")
```

## Features
1. Luas_Tanah (Land area in m²)
2. Luas_Bangunan (Building area in m²)
3. Jarak_ke_Pusat_Kota (Distance to city center in km)
4. Jumlah_Kamar_Tidur (Number of bedrooms)
5. Umur_Bangunan (Building age in years)

## Performance Metrics
- Cross-Validation R²: 0.9222 (±0.0576)
- Balance Score (Overfitting): 0.0083

## Files
- `best_property_price_model.pkl`: Trained model
- `best_poly_transformer.pkl`: Polynomial feature transformer (degree 2)
- `property_scaler.pkl`: Feature scaler
- `model_metadata.json`: Model metadata and information
