"""
Retrain phishing model using ONLY features extractable from URL string
(No DNS, WHOIS, HTTP requests required)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING MODEL - URL-ONLY FEATURES")
print("="*80)

# Load original data
df = pd.read_csv('dataset_small.csv')
print(f"\nOriginal data: {df.shape}")

# Define features that DON'T require network access
# These can be extracted from URL string alone
URL_ONLY_FEATURES = [
    # URL character counts (1-19)
    'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
    'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
    'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
    'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
    'qty_percent_url', 'qty_tld_url', 'length_url',
    
    # Domain features (20-40)
    'qty_dot_domain', 'qty_hyphen_domain', 'qty_underline_domain',
    'qty_slash_domain', 'qty_questionmark_domain', 'qty_equal_domain',
    'qty_at_domain', 'qty_and_domain', 'qty_exclamation_domain',
    'qty_space_domain', 'qty_tilde_domain', 'qty_comma_domain',
    'qty_plus_domain', 'qty_asterisk_domain', 'qty_hashtag_domain',
    'qty_dollar_domain', 'qty_percent_domain', 'qty_vowels_domain',
    'domain_length', 'domain_in_ip', 'server_client_domain',
    
    # Directory features (41-58)
    'qty_dot_directory', 'qty_hyphen_directory', 'qty_underline_directory',
    'qty_slash_directory', 'qty_questionmark_directory', 'qty_equal_directory',
    'qty_at_directory', 'qty_and_directory', 'qty_exclamation_directory',
    'qty_space_directory', 'qty_tilde_directory', 'qty_comma_directory',
    'qty_plus_directory', 'qty_asterisk_directory', 'qty_hashtag_directory',
    'qty_dollar_directory', 'qty_percent_directory', 'directory_length',
    
    # File features (59-76)
    'qty_dot_file', 'qty_hyphen_file', 'qty_underline_file',
    'qty_slash_file', 'qty_questionmark_file', 'qty_equal_file',
    'qty_at_file', 'qty_and_file', 'qty_exclamation_file',
    'qty_space_file', 'qty_tilde_file', 'qty_comma_file',
    'qty_plus_file', 'qty_asterisk_file', 'qty_hashtag_file',
    'qty_dollar_file', 'qty_percent_file', 'file_length',
    
    # Parameters features (77-96)
    'qty_dot_params', 'qty_hyphen_params', 'qty_underline_params',
    'qty_slash_params', 'qty_questionmark_params', 'qty_equal_params',
    'qty_at_params', 'qty_and_params', 'qty_exclamation_params',
    'qty_space_params', 'qty_tilde_params', 'qty_comma_params',
    'qty_plus_params', 'qty_asterisk_params', 'qty_hashtag_params',
    'qty_dollar_params', 'qty_percent_params', 'params_length',
    'tld_present_params', 'qty_params',
    
    # Simple features (97, 107, 111)
    'email_in_url',
    'tls_ssl_certificate',
    'url_shortened'
]

# EXCLUDED features (require network access):
EXCLUDED_FEATURES = [
    'time_response',           # HTTP request
    'domain_spf',              # DNS lookup
    'asn_ip',                  # IP/ASN lookup
    'time_domain_activation',  # WHOIS
    'time_domain_expiration',  # WHOIS
    'qty_ip_resolved',         # DNS lookup
    'qty_nameservers',         # DNS lookup
    'qty_mx_servers',          # DNS lookup
    'ttl_hostname',            # DNS lookup
    'qty_redirects',           # HTTP request
    'url_google_index',        # Google API
    'domain_google_index'      # Google API
]

print(f"\nUsing {len(URL_ONLY_FEATURES)} URL-parseable features")
print(f"Excluding {len(EXCLUDED_FEATURES)} network-dependent features")

# Prepare data
X = df[URL_ONLY_FEATURES]
y = df['phishing']

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples: {len(X):,}")
print(f"\nClass distribution:")
print(y.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# Train models
print("\n" + "="*80)
print("TRAINING MODELS (URL-ONLY FEATURES)")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\n{name}...")
    
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    gap = train_acc - test_acc
    
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
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | F1: {f1:.4f} | Gap: {gap:.4f}")

# Comparison
print("\n" + "="*80)
print("RESULTS")
print("="*80)

df_results = pd.DataFrame(results).sort_values('Test_Acc', ascending=False)
print("\n" + df_results[['Model', 'Train_Acc', 'Test_Acc', 'F1', 'Gap']].to_string(index=False))

# Best model
best = df_results.iloc[0]
best_model = models[best['Model']]

print(f"\n{'='*80}")
print(f"BEST: {best['Model']}")
print(f"  Test Accuracy: {best['Test_Acc']:.4f}")
print(f"  F1 Score: {best['F1']:.4f}")
print(f"  Overfit Gap: {best['Gap']:.4f}")
print("="*80)

# Classification report
y_pred = best_model.predict(X_test_scaled)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# Comparison with original model
print("\n" + "="*80)
print("COMPARISON WITH ORIGINAL MODEL")
print("="*80)
print(f"Original (111 features, 26% network-dependent): 94.35% accuracy")
print(f"URL-Only ({len(URL_ONLY_FEATURES)} features, 0% network-dependent): {best['Test_Acc']:.2%} accuracy")

accuracy_drop = 0.9435 - best['Test_Acc']
print(f"\nAccuracy drop: {accuracy_drop:.2%}")

if accuracy_drop < 0.05:
    print("✓ Minimal accuracy loss (<5%) - URL-only model is viable!")
elif accuracy_drop < 0.10:
    print("⚠️  Moderate accuracy loss (5-10%) - acceptable for deployment")
else:
    print("❌ High accuracy loss (>10%) - network features were critical")

# Save new model
print("\n" + "="*80)
print("SAVING URL-ONLY MODEL")
print("="*80)

joblib.dump(best_model, 'phishing_model_url_only.pkl')
joblib.dump(scaler, 'scaler_url_only.pkl')
joblib.dump(URL_ONLY_FEATURES, 'url_only_features.pkl')

print("✓ Model saved: phishing_model_url_only.pkl")
print("✓ Scaler saved: scaler_url_only.pkl")
print("✓ Features saved: url_only_features.pkl")

print("\n" + "="*80)
print("✓ DONE! Use url_predictor_final.py for predictions")
print("="*80)