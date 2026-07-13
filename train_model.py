import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure folders exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Starting GenMilk AI Model Training Pipeline...")

# ----------------------------------------------------
# 1. Load and Preprocess Cohort Data
# ----------------------------------------------------
try:
    df_m = pd.read_csv('data/HO_M.csv')
    df_f = pd.read_csv('data/HO_f.csv')
    df_p = pd.read_csv('data/HO_p.csv')
    print("Successfully loaded HO_M.csv, HO_f.csv, and HO_p.csv.")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    # Create mock data pathways if data is missing, but here the files exist
    raise e

# Verify year column and sort
df_m = df_m.sort_values(by='yob1').reset_index(drop=True)
df_f = df_f.sort_values(by='yob1').reset_index(drop=True)
df_p = df_p.sort_values(by='yob1').reset_index(drop=True)

# Interpolate / fill missing values in the cohort datasets to ensure smooth curves
df_m = df_m.ffill().bfill()
df_f = df_f.ffill().bfill()
df_p = df_p.ffill().bfill()

# ----------------------------------------------------
# 2. Simulate Individual-Level Cow Population (N = 5000)
# ----------------------------------------------------
print("Generating simulated individual cattle population based on genomic cohort statistics...")
np.random.seed(42)

cows = []
years = df_m['yob1'].unique()
cows_per_year = 120  # ~4920 cows total for 41 cohorts

for year in years:
    # Get mean and standard deviations from cohort data for the given year
    row_m = df_m[df_m['yob1'] == year].iloc[0]
    row_f = df_f[df_f['yob1'] == year].iloc[0]
    row_p = df_p[df_p['yob1'] == year].iloc[0]
    
    # Cohort means (from registered cow columns)
    mean_ptam = row_m['HO_ptam_RegCow_1']
    mean_ptaf = row_f['HO_ptaf_RegCow_1']
    mean_ptap = row_p['HO_ptap_RegCow_1']
    
    # Cohort Standard Deviations (use 'W_BY1_HO_SD' SC columns if available and valid (>0), otherwise defaults)
    sd_ptam = row_m['W_BY1_HO_SD_m_SC'] if ('W_BY1_HO_SD_m_SC' in row_m and row_m['W_BY1_HO_SD_m_SC'] > 0) else 450.0
    sd_ptaf = row_f['W_BY1_HO_SD_f_SC'] if ('W_BY1_HO_SD_f_SC' in row_f and row_f['W_BY1_HO_SD_f_SC'] > 0) else 18.0
    sd_ptap = row_p['W_BY1_HO_SD_p_SC'] if ('W_BY1_HO_SD_p_SC' in row_p and row_p['W_BY1_HO_SD_p_SC'] > 0) else 12.0
    
    # Fallback to defaults if SD values are NaN
    if np.isnan(sd_ptam): sd_ptam = 450.0
    if np.isnan(sd_ptaf): sd_ptaf = 18.0
    if np.isnan(sd_ptap): sd_ptap = 12.0
    
    # Generate individuals for this birth year
    for _ in range(cows_per_year):
        # Sample genomic traits from Normal distributions
        ptam = np.random.normal(mean_ptam, sd_ptam)
        ptaf = np.random.normal(mean_ptaf, sd_ptaf)
        ptap = np.random.normal(mean_ptap, sd_ptap)
        
        # Simulate health and fertility parameters with biological trends
        # 1. Productive Life (ptapl) - shows moderate progress over time
        mean_ptapl = -1.5 + 4.5 * ((year - 1975) / 40.0)
        ptapl = np.random.normal(mean_ptapl, 1.8)
        
        # 2. Daughter Pregnancy Rate (ptadpr) - historically declined with extreme milk selection, then recovered
        mean_ptadpr = 1.0 - 2.5 * ((year - 1975) / 40.0)
        ptadpr = np.random.normal(mean_ptadpr, 1.3)
        
        # 3. Somatic Cell Score (ptascs) - lower is better (less mastitis risk). Bred for lower values over time.
        mean_ptascs = 3.2 - 0.35 * ((year - 1975) / 40.0)
        ptascs = np.random.normal(mean_ptascs, 0.22)
        
        # 4. Reliability Score (REL) - uniform genomic test reliability
        rel = np.random.uniform(70.0, 98.0)
        
        # Calculate Target Production Outputs (with historic trends, genetic contributions, and environmental noise)
        # Milk yield (lbs per lactation): base grows from ~15,000 lbs in 1975 to ~25,000 lbs in 2015
        base_milk_yield = 16000.0 + 8500.0 * ((year - 1975) / 40.0)
        milk_yield = base_milk_yield + (2.0 * ptam) + np.random.normal(0.0, 950.0)
        
        # Fat yield (lbs): base grows from ~600 to ~950
        base_fat_yield = 600.0 + 340.0 * ((year - 1975) / 40.0)
        fat_yield = base_fat_yield + (2.0 * ptaf) + np.random.normal(0.0, 35.0)
        
        # Protein yield (lbs): base grows from ~480 to ~760
        base_protein_yield = 480.0 + 260.0 * ((year - 1975) / 40.0)
        protein_yield = base_protein_yield + (2.0 * ptap) + np.random.normal(0.0, 25.0)
        
        # Genetic Potential Score Formula (0-100)
        # Scale each individual value to relative breed range for scoring
        s_m = np.clip((ptam + 2500.0) / 3800.0, 0.0, 1.0)
        s_f = np.clip((ptaf + 100.0) / 160.0, 0.0, 1.0)
        s_p = np.clip((ptap + 100.0) / 160.0, 0.0, 1.0)
        s_pl = np.clip((ptapl + 5.0) / 15.0, 0.0, 1.0)
        s_dpr = np.clip((ptadpr + 5.0) / 10.0, 0.0, 1.0)
        s_scs = np.clip((4.5 - ptascs) / 2.5, 0.0, 1.0) # SCS is inverted, lower is better
        
        s_prod = 0.4 * s_m + 0.3 * s_f + 0.3 * s_p
        s_health = 0.4 * s_pl + 0.3 * s_dpr + 0.3 * s_scs
        gps = 100.0 * (0.55 * s_prod + 0.45 * s_health)
        gps = float(np.round(np.clip(gps, 0.0, 100.0), 2))
        
        # Category classification (based on relative milk yield)
        # Thresholds: Low < 19,000 lbs, High >= 24,000 lbs, Medium in between
        if milk_yield < 19200:
            category = "Low Producer"
        elif milk_yield >= 24200:
            category = "High Producer"
        else:
            category = "Medium Producer"
            
        cows.append({
            'yob': int(year),
            'ptam': ptam,
            'ptaf': ptaf,
            'ptap': ptap,
            'ptapl': ptapl,
            'ptadpr': ptadpr,
            'ptascs': ptascs,
            'REL': rel,
            'milk_yield': milk_yield,
            'fat_yield': fat_yield,
            'protein_yield': protein_yield,
            'gps': gps,
            'category': category
        })

df_sim = pd.DataFrame(cows)
print(f"Generated dataset with {len(df_sim)} dairy cattle records.")
print(df_sim.head())

# Save simulated dataset for audit
df_sim.to_csv('data/simulated_cattle_data.csv', index=False)

# ----------------------------------------------------
# 3. Model Preparation, Scaling, and Splitting
# ----------------------------------------------------
features = ['ptam', 'ptaf', 'ptap', 'ptapl', 'ptadpr', 'ptascs', 'REL', 'yob']
targets = ['milk_yield', 'fat_yield', 'protein_yield']

X = df_sim[features]
y = df_sim[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler using standard pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved standard scaler as models/scaler.pkl")

# ----------------------------------------------------
# 4. Model Training & Evaluation (R2, MAE, RMSE)
# ----------------------------------------------------
metrics = {}
trained_models = {}

for target in targets:
    print(f"\nTraining Random Forest Regressor for '{target}'...")
    
    # Initialize Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit model on scaled features
    model.fit(X_train_scaled, y_train[target])
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    r2 = r2_score(y_test[target], y_pred)
    mae = mean_absolute_error(y_test[target], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
    
    print(f"  R² Score:  {r2:.5f}")
    print(f"  MAE:       {mae:.2f}")
    print(f"  RMSE:      {rmse:.2f}")
    
    # Get feature importance
    importance = model.feature_importances_
    feat_importance = dict(zip(features, importance))
    
    metrics[target] = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'feature_importance': feat_importance
    }
    
    trained_models[target] = model
    
    # Save the model using standard pickle
    model_path = f'models/model_{target.split("_")[0]}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")

# Save the metrics dictionary using standard pickle
with open('models/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("\nAll models and scalers have been successfully trained and saved!")
