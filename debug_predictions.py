import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('phishing_model.pkl')

# Check if model has feature importance
if hasattr(model, 'feature_importances_'):
    scaler = joblib.load('scaler.pkl')
    feature_names = scaler.feature_names_in_
    importances = model.feature_importances_
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("="*80)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*80)
    print(importance_df.head(20).to_string(index=False))
    
    print("\n" + "="*80)
    print("FEATURES WE'RE SETTING TO 0 (potentially important)")
    print("="*80)
    
    # Features we default to 0
    default_zero_features = [
        'time_response', 'domain_spf', 'asn_ip', 
        'time_domain_activation', 'time_domain_expiration',
        'qty_nameservers', 'qty_mx_servers', 'ttl_hostname',
        'qty_redirects', 'url_google_index', 'domain_google_index'
    ]
    
    for feat in default_zero_features:
        if feat in importance_df['Feature'].values:
            imp = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
            if imp > 0.01:  # If importance > 1%
                print(f"⚠️  {feat}: {imp:.4f} (HIGH IMPORTANCE - but we set to 0!)")
            else:
                print(f"✓ {feat}: {imp:.4f} (low importance)")
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    # Check cumulative importance of zero-defaulted features
    zero_feat_importance = importance_df[
        importance_df['Feature'].isin(default_zero_features)
    ]['Importance'].sum()
    
    print(f"\nTotal importance of zero-defaulted features: {zero_feat_importance:.2%}")
    
    if zero_feat_importance > 0.20:
        print("❌ CRITICAL: Over 20% of model decisions rely on missing features!")
        print("   This explains the poor predictions.")
    elif zero_feat_importance > 0.10:
        print("⚠️  WARNING: 10-20% of model relies on missing features")
    else:
        print("✓ Missing features have low impact (<10%)")
        
else:
    print("Model doesn't have feature_importances_ attribute")
    print("Model type:", type(model))

# Load training data to check feature distributions
print("\n" + "="*80)
print("ANALYZING TRAINING DATA PATTERNS")
print("="*80)

df = pd.read_csv('dataset_small.csv')

# Check how often features are -1 or 0 in training
print("\nFeatures that are often -1 in training data:")
for col in df.columns[:-1]:  # Exclude target
    pct_minus_one = (df[col] == -1).sum() / len(df) * 100
    pct_zero = (df[col] == 0).sum() / len(df) * 100
    if pct_minus_one > 50:
        print(f"  {col}: {pct_minus_one:.1f}% are -1")
    elif pct_zero > 80:
        print(f"  {col}: {pct_zero:.1f}% are 0")