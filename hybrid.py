"""
Standalone Hybrid Phishing Detector
Combines domain reputation rules + ML model
All code in one file - no external imports needed
"""

import re
from urllib.parse import urlparse
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


# =================================================================
# CONFIGURATION
# =================================================================

# Known legitimate domains
KNOWN_LEGITIMATE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'linkedin.com', 'github.com', 'stackoverflow.com', 'reddit.com', 'wikipedia.org',
    'amazon.com', 'ebay.com', 'walmart.com', 'target.com', 'bestbuy.com',
    'apple.com', 'microsoft.com', 'ibm.com', 'oracle.com', 'adobe.com',
    'netflix.com', 'spotify.com', 'twitch.tv', 'discord.com', 'zoom.us',
    'gmail.com', 'outlook.com', 'yahoo.com', 'protonmail.com',
    'paypal.com', 'stripe.com', 'square.com', 'venmo.com',
    'dropbox.com', 'onedrive.com', 'icloud.com', 'box.com',
    'wordpress.com', 'blogger.com', 'medium.com', 'tumblr.com',
    'cnn.com', 'bbc.com', 'nytimes.com', 'washingtonpost.com',
    'nih.gov', 'cdc.gov', 'who.int', 'harvard.edu', 'mit.edu',
}

TYPOSQUATTING_PATTERNS = [
    'paypa1', 'g00gle', 'micros0ft', 'faceb00k', 'youtu6e',
    'amaz0n', 'app1e', 'netf1ix', 'yah00', 'tw1tter',
]

SUSPICIOUS_KEYWORDS = [
    'verify', 'account', 'update', 'confirm', 'suspend', 'secure',
    'signin', 'login', 'banking', 'alert', 'warning', 'unlock',
    'validate', 'restore', 'limited', 'unusual', 'activity',
    'free-', 'winner', 'prize', 'claim', 'gift-card', 'urgent',
]

URL_SHORTENERS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 
    'is.gd', 'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in',
}


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def count_vowels(text):
    """Count vowels in text"""
    return sum(1 for char in text.lower() if char in 'aeiou')


def has_client_server_words(domain):
    """Check if domain contains client/server keywords"""
    keywords = ['client', 'server', 'admin', 'host', 'user']
    return 1 if any(word in domain.lower() for word in keywords) else 0


def get_base_domain(domain):
    """Extract base domain (e.g., github.com from www.github.com)"""
    parts = domain.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return domain


def check_typosquatting(domain):
    """Check if domain looks like typo-squatting"""
    domain_lower = domain.lower()
    for pattern in TYPOSQUATTING_PATTERNS:
        if pattern in domain_lower:
            return True
    return False


def count_suspicious_keywords(url):
    """Count suspicious keywords in URL"""
    url_lower = url.lower()
    return sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in url_lower)


def is_ip_address(domain):
    """Check if domain is an IP address"""
    return bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$', domain))


def extract_url_features(url):
    """
    Extract 99 URL-only features
    """
    features = {}
    parsed = urlparse(url)
    
    domain = parsed.netloc
    directory = parsed.path.rsplit('/', 1)[0] if '/' in parsed.path else ''
    file_part = parsed.path.rsplit('/', 1)[1] if '/' in parsed.path else ''
    params = parsed.query
    
    # URL character counts (1-19)
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
    features['qty_tld_url'] = float(url.count('.'))
    features['length_url'] = float(len(url))
    
    # Domain features (20-40)
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
    
    ip_check = bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain))
    features['domain_in_ip'] = float(1 if ip_check else 0)
    features['server_client_domain'] = float(has_client_server_words(domain))
    
    # Directory features (41-58)
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
    
    # File features (59-76)
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
    
    # Parameters features (77-96)
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
    features['tld_present_params'] = -1.0
    features['qty_params'] = float(len(params.split('&'))) if has_params else -1.0
    
    # Simple features (97-99)
    features['email_in_url'] = float(1 if '@' in url else 0)
    features['tls_ssl_certificate'] = float(1 if url.startswith('https://') else 0)
    
    shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly']
    features['url_shortened'] = float(1 if any(s in domain for s in shorteners) else 0)
    
    return features


# =================================================================
# HYBRID DETECTOR
# =================================================================

class HybridPhishingDetector:
    """
    Hybrid detector: Rules + ML model
    """
    
    def __init__(self, model_path='phishing_model_url_only.pkl',
                 scaler_path='scaler_url_only.pkl'):
        """Initialize"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("✓ Hybrid phishing detector loaded")
        print(f"  - {len(KNOWN_LEGITIMATE_DOMAINS)} whitelisted domains")
        print(f"  - {len(TYPOSQUATTING_PATTERNS)} typo patterns")
    
    def predict(self, url, explain=False):
        """Predict if URL is phishing"""
        parsed = urlparse(url)
        domain = parsed.netloc
        base_domain = get_base_domain(domain)
        
        # RULE 1: Whitelist
        if base_domain in KNOWN_LEGITIMATE_DOMAINS:
            if explain:
                return 'LEGITIMATE', 0.99, f"Whitelisted: {base_domain}"
            return 'LEGITIMATE', 0.99
        
        # RULE 2: Typo-squatting
        if check_typosquatting(domain):
            if explain:
                return 'PHISHING', 0.95, "Typo-squatting detected"
            return 'PHISHING', 0.95
        
        # RULE 3: IP address (non-local)
        if is_ip_address(domain):
            if not (domain.startswith('192.168.') or domain.startswith('10.') or domain.startswith('127.')):
                if explain:
                    return 'PHISHING', 0.80, "Public IP address"
                return 'PHISHING', 0.80
        
        # RULE 4: URL shortener
        if base_domain in URL_SHORTENERS:
            if explain:
                return 'SUSPICIOUS', 0.50, f"URL shortener: {base_domain}"
            return 'SUSPICIOUS', 0.50
        
        # RULE 5: Suspicious keywords
        keyword_count = count_suspicious_keywords(url)
        if keyword_count >= 3:
            if explain:
                return 'PHISHING', 0.85, f"{keyword_count} suspicious keywords"
            return 'PHISHING', 0.85
        
        # RULE 6: ML Model
        features = extract_url_features(url)
        df = pd.DataFrame([features])
        X_scaled = self.scaler.transform(df)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        label = 'PHISHING' if prediction == 1 else 'LEGITIMATE'
        confidence = probabilities[prediction]
        
        # Boost confidence if keywords present
        if keyword_count >= 2 and label == 'PHISHING':
            confidence = min(0.95, confidence + 0.10)
        
        if explain:
            reason = f"ML model ({confidence:.1%})"
            if keyword_count > 0:
                reason += f" + {keyword_count} suspicious keywords"
            return label, confidence, reason
        
        return label, confidence


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    print("="*80)
    print("HYBRID PHISHING DETECTOR")
    print("="*80)
    
    detector = HybridPhishingDetector()
    
    test_urls = [
        'https://www.google.com',
        'https://www.github.com/user/repo',
        'http://paypal-verify-account.suspicious-domain.com/login.php',
        'http://192.168.1.1/admin',
        'https://bit.ly/abc123',
        'http://secure-login-apple-id-verify.com/signin.php?user=test',
        'https://www.amazon.com/products/item',
        'http://www.paypa1.com',
        'https://www.microsoft.com',
        'http://free-iphone-winner-claim-now.com',
    ]
    
    print("\n" + "="*80)
    print("TESTING URLS")
    print("="*80)
    
    for i, url in enumerate(test_urls, 1):
        label, conf, reason = detector.predict(url, explain=True)
        
        if label == 'PHISHING':
            status = "🚨"
        elif label == 'SUSPICIOUS':
            status = "⚠️ "
        else:
            status = "✓"
        
        print(f"\n[{i}] {url}")
        print(f"    {status} {label} ({conf:.1%}) - {reason}")
    
    print("\n" + "="*80)
    print("✓ DONE!")
    print("="*80)