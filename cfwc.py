import joblib
import pandas as pd

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Get the feature names the model was trained on
feature_names = scaler.feature_names_in_

print("="*80)
print(f"YOUR MODEL EXPECTS {len(feature_names)} FEATURES")
print("="*80)

print("\nFeature names (in exact order):")
print("-"*80)
for i, feat in enumerate(feature_names, 1):
    print(f"{i:3d}. {feat}")

# Save to file
with open('required_features.txt', 'w') as f:
    f.write("Required features for prediction:\n\n")
    for i, feat in enumerate(feature_names, 1):
        f.write(f"{i}. {feat}\n")

print("\n" + "="*80)
print("✓ Feature list saved to 'required_features.txt'")
print("="*80)

# Also load a sample from the training data to see values
print("\nNow checking your original training data...")
try:
    df = pd.read_csv('dataset_small.csv')
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"\nFirst row sample:")
    print("-"*80)
    
    # Show first row of features (excluding target)
    first_row = df.iloc[0, :-1]
    for feat, val in first_row.items():
        print(f"{feat}: {val}")
    
    print("\nSaving sample to 'sample_features.csv'...")
    df.iloc[0:1, :-1].to_csv('sample_features.csv', index=False)
    print("✓ Saved!")
    
except Exception as e:
    print(f"Could not load dataset: {e}")