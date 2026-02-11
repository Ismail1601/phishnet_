"""
URL Feature Extractor for Phishing Detection
Extracts exactly 111 features matching dataset_small.csv format
"""

import re
from urllib.parse import urlparse
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


def count_vowels(text):
    """Count vowels in text"""
    return sum(1 for char in text.lower() if char in 'aeiou')


def has_client_server_words(domain):
    """Check if domain contains client/server keywords"""
    keywords = ['client', 'server', 'admin', 'host', 'user']
    return 1 if any(word in domain.lower() for word in keywords) else 0


def extract_url_features(url):
    """
    Extract 111 features from a URL string matching the training data format
    
    Parameters:
    -----------
    url : str
        Raw URL string (e.g., "https://example.com/path?query=1")
    
    Returns:
    --------
    features : dict (111 features in exact order)
    """
    
    features = {}
    
    # Parse URL
    parsed = urlparse(url)
    
    # Get components
    domain = parsed.netloc
    directory = parsed.path.rsplit('/', 1)[0] if '/' in parsed.path else ''
    file_part = parsed.path.rsplit('/', 1)[1] if '/' in parsed.path else ''
    params = parsed.query
    
    # =================================================================
    # 1-18: URL character counts
    # =================================================================
    features['qty_dot_url'] = float(url.count('.'))
    features['qty_hyphen_url'] = float(url.count('-'))
    features['qty_underline_url'] = float(url.count('_'))
    features['qty_slash_url'] = float(url.count('/'))
    features['qty_questionmark_url'] = float(url.count('?'))
    features['qty_equal_url'] = float(url.count('='))
    features['qty_at_url'] = float(url.count('@'))
    features['qty_and_url'] = float(url.count('&'))
    features['qty_exclamation_url'] = float(url.count('!'))
    features['qty_space_url'] = float(url.count(' '))
    features['qty_tilde_url'] = float(url.count('~'))
    features['qty_comma_url'] = float(url.count(','))
    features['qty_plus_url'] = float(url.count('+'))
    features['qty_asterisk_url'] = float(url.count('*'))
    features['qty_hashtag_url'] = float(url.count('#'))
    features['qty_dollar_url'] = float(url.count('$'))
    features['qty_percent_url'] = float(url.count('%'))
    
    # 18-19: TLD and length
    features['qty_tld_url'] = float(url.count('.'))  # Approximation
    features['length_url'] = float(len(url))
    
    # =================================================================
    # 20-40: Domain features
    # =================================================================
    features['qty_dot_domain'] = float(domain.count('.'))
    features['qty_hyphen_domain'] = float(domain.count('-'))
    features['qty_underline_domain'] = float(domain.count('_'))
    features['qty_slash_domain'] = float(domain.count('/'))
    features['qty_questionmark_domain'] = float(domain.count('?'))
    features['qty_equal_domain'] = float(domain.count('='))
    features['qty_at_domain'] = float(domain.count('@'))
    features['qty_and_domain'] = float(domain.count('&'))
    features['qty_exclamation_domain'] = float(domain.count('!'))
    features['qty_space_domain'] = float(domain.count(' '))
    features['qty_tilde_domain'] = float(domain.count('~'))
    features['qty_comma_domain'] = float(domain.count(','))
    features['qty_plus_domain'] = float(domain.count('+'))
    features['qty_asterisk_domain'] = float(domain.count('*'))
    features['qty_hashtag_domain'] = float(domain.count('#'))
    features['qty_dollar_domain'] = float(domain.count('$'))
    features['qty_percent_domain'] = float(domain.count('%'))
    features['qty_vowels_domain'] = float(count_vowels(domain))
    features['domain_length'] = float(len(domain))
    
    # Check if domain is IP
    is_ip = bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain))
    features['domain_in_ip'] = float(1 if is_ip else 0)
    features['server_client_domain'] = float(has_client_server_words(domain))
    
    # =================================================================
    # 41-58: Directory features (use -1 if no directory)
    # =================================================================
    has_directory = len(directory) > 0
    features['qty_dot_directory'] = float(directory.count('.')) if has_directory else -1.0
    features['qty_hyphen_directory'] = float(directory.count('-')) if has_directory else -1.0
    features['qty_underline_directory'] = float(directory.count('_')) if has_directory else -1.0
    features['qty_slash_directory'] = float(directory.count('/')) if has_directory else -1.0
    features['qty_questionmark_directory'] = float(directory.count('?')) if has_directory else -1.0
    features['qty_equal_directory'] = float(directory.count('=')) if has_directory else -1.0
    features['qty_at_directory'] = float(directory.count('@')) if has_directory else -1.0
    features['qty_and_directory'] = float(directory.count('&')) if has_directory else -1.0
    features['qty_exclamation_directory'] = float(directory.count('!')) if has_directory else -1.0
    features['qty_space_directory'] = float(directory.count(' ')) if has_directory else -1.0
    features['qty_tilde_directory'] = float(directory.count('~')) if has_directory else -1.0
    features['qty_comma_directory'] = float(directory.count(',')) if has_directory else -1.0
    features['qty_plus_directory'] = float(directory.count('+')) if has_directory else -1.0
    features['qty_asterisk_directory'] = float(directory.count('*')) if has_directory else -1.0
    features['qty_hashtag_directory'] = float(directory.count('#')) if has_directory else -1.0
    features['qty_dollar_directory'] = float(directory.count('$')) if has_directory else -1.0
    features['qty_percent_directory'] = float(directory.count('%')) if has_directory else -1.0
    features['directory_length'] = float(len(directory)) if has_directory else -1.0
    
    # =================================================================
    # 59-76: File features (use -1 if no file)
    # =================================================================
    has_file = len(file_part) > 0
    features['qty_dot_file'] = float(file_part.count('.')) if has_file else -1.0
    features['qty_hyphen_file'] = float(file_part.count('-')) if has_file else -1.0
    features['qty_underline_file'] = float(file_part.count('_')) if has_file else -1.0
    features['qty_slash_file'] = float(file_part.count('/')) if has_file else -1.0
    features['qty_questionmark_file'] = float(file_part.count('?')) if has_file else -1.0
    features['qty_equal_file'] = float(file_part.count('=')) if has_file else -1.0
    features['qty_at_file'] = float(file_part.count('@')) if has_file else -1.0
    features['qty_and_file'] = float(file_part.count('&')) if has_file else -1.0
    features['qty_exclamation_file'] = float(file_part.count('!')) if has_file else -1.0
    features['qty_space_file'] = float(file_part.count(' ')) if has_file else -1.0
    features['qty_tilde_file'] = float(file_part.count('~')) if has_file else -1.0
    features['qty_comma_file'] = float(file_part.count(',')) if has_file else -1.0
    features['qty_plus_file'] = float(file_part.count('+')) if has_file else -1.0
    features['qty_asterisk_file'] = float(file_part.count('*')) if has_file else -1.0
    features['qty_hashtag_file'] = float(file_part.count('#')) if has_file else -1.0
    features['qty_dollar_file'] = float(file_part.count('$')) if has_file else -1.0
    features['qty_percent_file'] = float(file_part.count('%')) if has_file else -1.0
    features['file_length'] = float(len(file_part)) if has_file else -1.0
    
    # =================================================================
    # 77-96: Parameters features (use -1 if no params)
    # =================================================================
    has_params = len(params) > 0
    features['qty_dot_params'] = float(params.count('.')) if has_params else -1.0
    features['qty_hyphen_params'] = float(params.count('-')) if has_params else -1.0
    features['qty_underline_params'] = float(params.count('_')) if has_params else -1.0
    features['qty_slash_params'] = float(params.count('/')) if has_params else -1.0
    features['qty_questionmark_params'] = float(params.count('?')) if has_params else -1.0
    features['qty_equal_params'] = float(params.count('=')) if has_params else -1.0
    features['qty_at_params'] = float(params.count('@')) if has_params else -1.0
    features['qty_and_params'] = float(params.count('&')) if has_params else -1.0
    features['qty_exclamation_params'] = float(params.count('!')) if has_params else -1.0
    features['qty_space_params'] = float(params.count(' ')) if has_params else -1.0
    features['qty_tilde_params'] = float(params.count('~')) if has_params else -1.0
    features['qty_comma_params'] = float(params.count(',')) if has_params else -1.0
    features['qty_plus_params'] = float(params.count('+')) if has_params else -1.0
    features['qty_asterisk_params'] = float(params.count('*')) if has_params else -1.0
    features['qty_hashtag_params'] = float(params.count('#')) if has_params else -1.0
    features['qty_dollar_params'] = float(params.count('$')) if has_params else -1.0
    features['qty_percent_params'] = float(params.count('%')) if has_params else -1.0
    features['params_length'] = float(len(params)) if has_params else -1.0
    features['tld_present_params'] = -1.0  # Complex to determine
    features['qty_params'] = float(len(params.split('&'))) if has_params else -1.0
    
    # =================================================================
    # 97-111: Advanced features (set to default values)
    # =================================================================
    features['email_in_url'] = float(1 if '@' in url else 0)
    features['time_response'] = 0.0  # Requires HTTP request
    features['domain_spf'] = 0.0     # Requires DNS lookup
    features['asn_ip'] = 0.0         # Requires IP lookup
    features['time_domain_activation'] = 0.0  # Requires WHOIS
    features['time_domain_expiration'] = 0.0  # Requires WHOIS
    features['qty_ip_resolved'] = float(1 if is_ip else 0)
    features['qty_nameservers'] = 0.0  # Requires DNS lookup
    features['qty_mx_servers'] = 0.0   # Requires DNS lookup
    features['ttl_hostname'] = 0.0     # Requires DNS lookup
    features['tls_ssl_certificate'] = float(1 if url.startswith('https://') else 0)
    features['qty_redirects'] = 0.0    # Requires HTTP request
    features['url_google_index'] = 0.0 # Requires Google API
    features['domain_google_index'] = 0.0  # Requires Google API
    
    # Check if URL is shortened
    shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 
                  'is.gd', 'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in']
    features['url_shortened'] = float(1 if any(short in domain for short in shorteners) else 0)
    
    return features


class PhishingDetector:
    """
    Phishing URL detector using trained model
    """
    
    def __init__(self, model_path='phishing_model.pkl', scaler_path='scaler.pkl'):
        """Load trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("✓ Model and scaler loaded successfully")
    
    def predict(self, url):
        """
        Predict if URL is phishing or legitimate
        
        Parameters:
        -----------
        url : str
            Raw URL string
        
        Returns:
        --------
        prediction : str
            'PHISHING' or 'LEGITIMATE'
        confidence : float
            Confidence score (0-1)
        """
        # Extract features
        features = extract_url_features(url)
        
        # Convert to DataFrame with correct column order
        df = pd.DataFrame([features])
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        label = 'PHISHING' if prediction == 1 else 'LEGITIMATE'
        confidence = probabilities[prediction]
        
        return label, confidence
    
    def predict_batch(self, urls):
        """Predict multiple URLs at once"""
        results = []
        for url in urls:
            try:
                label, conf = self.predict(url)
                results.append({
                    'url': url, 
                    'prediction': label, 
                    'confidence': f"{conf:.2%}"
                })
            except Exception as e:
                results.append({
                    'url': url, 
                    'prediction': 'ERROR', 
                    'confidence': str(e)
                })
        return pd.DataFrame(results)


# =================================================================
# EXAMPLE USAGE
# =================================================================
if __name__ == "__main__":
    print("="*80)
    print("PHISHING URL DETECTOR - READY TO USE")
    print("="*80)
    
    # Initialize detector
    detector = PhishingDetector(
        model_path='phishing_model.pkl',
        scaler_path='scaler.pkl'
    )
    
    # Test URLs
    test_urls = [
        'https://www.google.com',
        'https://www.github.com/user/repo',
        'http://paypal-verify-account.suspicious-domain.com/login.php',
        'http://192.168.1.1/admin',
        'https://bit.ly/abc123',
        'http://secure-login-apple-id-verify.com/signin.php?user=test&redirect=home',
        'https://www.amazon.com/products/item',
        'http://www.paypa1.com',  # Note the '1' instead of 'l'
    ]
    
    print("\n" + "="*80)
    print("TESTING URLS")
    print("="*80)
    
    for i, url in enumerate(test_urls, 1):
        label, confidence = detector.predict(url)
        
        # Color coding for display
        status = "🚨" if label == "PHISHING" else "✓"
        
        print(f"\n[{i}] {url}")
        print(f"    {status} {label} (Confidence: {confidence:.1%})")
    
    print("\n" + "="*80)
    print("BATCH PREDICTION EXAMPLE")
    print("="*80)
    
    results_df = detector.predict_batch(test_urls)
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ Ready for production use!")
    print("="*80)
    
    print("\nUsage:")
    print("------")
    print("from url_predictor import PhishingDetector")
    print("")
    print("detector = PhishingDetector('phishing_model.pkl', 'scaler.pkl')")
    print("label, confidence = detector.predict('http://suspicious-url.com')")
    print("print(f'{label}: {confidence:.2%}')")