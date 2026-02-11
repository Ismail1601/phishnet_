import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHISHING DETECTION - DATASET_SMALL.CSV")
print("="*80)

# 1. Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('dataset_small.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.shape[1]}")

# 2. Identify target
print("\n[2/6] Target identification...")
target_col = df.columns[-1]
print(f"Target column: '{target_col}'")
print(f"Class distribution:\n{df[target_col].value_counts()}")

# 3. Prepare data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(f"\n[3/6] Data prepared")
print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# 4. Check for leakage (test a few features)
print("\n[4/6] Quick leakage check...")
X_temp, X_test_temp, y_temp, y_test_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

leaky = []
for col in X.columns[:15]:  # Check first 15 features
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_temp[[col]], y_temp)
    score = clf.score(X_test_temp[[col]], y_test_temp)
    if score > 0.85:
        leaky.append((col, score))

if leaky:
    print(f"⚠️  {len(leaky)} features with >85% single-feature accuracy:")
    for feat, acc in leaky[:3]:
        print(f"   - {feat}: {acc:.2%}")
else:
    print("✓ No obvious leakage detected")

# 5. Split and scale
print("\n[5/6] Training models...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_split=10, 
        min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, 
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, random_state=42
    )
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    gap = train_acc - test_acc
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results.append({
        'Model': name,
        'Train_Acc': train_acc,
        'Test_Acc': test_acc,
        'F1': f1,
        'Gap': gap,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    })
    
    print(f"\n{name}:")
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | F1: {f1:.4f}")
    print(f"  Gap: {gap:.4f}")
    print(f"  CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Results summary
print("\n" + "="*80)
print("[6/6] COMPARISON")
print("="*80)

df_results = pd.DataFrame(results).sort_values('Test_Acc', ascending=False)
print("\n" + df_results[['Model', 'Train_Acc', 'Test_Acc', 'F1', 'Gap']].to_string(index=False))

# Best model
best = df_results.iloc[0]
print(f"\n{'='*80}")
print(f"BEST: {best['Model']}")
print(f"  Test Accuracy: {best['Test_Acc']:.4f}")
print(f"  F1 Score: {best['F1']:.4f}")
print(f"  Overfit Gap: {best['Gap']:.4f}")
print("="*80)

# Get best model object
best_name = best['Model']
best_model = models[best_name]

# Classification report
print(f"\nClassification Report - {best_name}:")
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, digits=4))

# Assessment
print("\n" + "="*80)
print("ASSESSMENT")
print("="*80)

if best['Test_Acc'] > 0.98:
    print("⚠️  >98% accuracy - possible synthetic dataset or data leakage")
elif best['Test_Acc'] > 0.92:
    print("✓ Excellent! (92-98% is realistic for phishing detection)")
elif best['Test_Acc'] > 0.85:
    print("✓ Good performance (85-92% is acceptable)")
else:
    print("⚠️  Lower performance - may need feature engineering")

if best['Gap'] > 0.10:
    print("⚠️  High overfitting (>10% gap)")
elif best['Gap'] > 0.05:
    print("⚠️  Moderate overfitting (5-10% gap)")
else:
    print("✓ Low overfitting (<5% gap)")

# Save
print("\n" + "="*80)
print("SAVING")
print("="*80)

joblib.dump(best_model, 'phishing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✓ Model saved: phishing_model.pkl")
print("✓ Scaler saved: scaler.pkl")
print("\nDone!")