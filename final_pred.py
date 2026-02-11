"""
Hybrid Phishing Detector
Combines ML model with domain reputation rules
"""

import re
from urllib.parse import urlparse
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import the feature extraction from previous script
from phishing_detector_final import extract_url_features


# Known legitimate domains (top 1000 would be better)
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

# Typo-squatting patterns (common misspellings of popular sites)
TYPOSQUATTING_PATTERNS = [
    'paypa1', 'g00gle', 'micros0ft', 'faceb00k', 'youtu6e',
    'amaz0n', 'app1e', 'netf1ix', 'yah00', 'tw1tter',
]

# Suspicious keywords in domain/path
SUSPICIOUS_KEYWORDS = [
    'verify', 'account', 'update', 'confirm', 'suspend', 'secure',
    'signin', 'login', 'banking', 'alert', 'warning', 'unlock',
    'validate', 'restore', 'limited', 'unusual', 'activity',
    'free-', 'winner', 'prize', 'claim', 'gift-card', 'urgent',
]

# URL shortener services (treat with caution)
URL_SHORTENERS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 
    'is.gd', 'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in',
    'tiny.cc', 'rebrand.ly', 'short.io'
}


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


class HybridPhishingDetector:
    """
    Hybrid phishing detector using:
    1. Domain reputation (whitelist)
    2. Pattern-based rules (typo-squatting, suspicious keywords)
    3. ML model as fallback
    """
    
    def __init__(self, model_path='phishing_model_url_only.pkl',
                 scaler_path='scaler_url_only.pkl'):
        """Initialize with ML model"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("✓ Hybrid phishing detector loaded")
        print(f"  - {len(KNOWN_LEGITIMATE_DOMAINS)} whitelisted domains")
        print(f"  - {len(TYPOSQUATTING_PATTERNS)} typo patterns")
        print(f"  - ML model (90% accuracy)")
    
    def predict(self, url, explain=False):
        """
        Predict if URL is phishing
        
        Parameters:
        -----------
        url : str
            Raw URL
        explain : bool
            If True, return explanation of decision
        
        Returns:
        --------
        prediction : str
            'PHISHING' or 'LEGITIMATE'
        confidence : float
            Confidence (0-1)
        reason : str (if explain=True)
            Why the decision was made
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        base_domain = get_base_domain(domain)
        
        reasons = []
        
        # ==============================================================
        # RULE 1: Whitelist check (high confidence legitimate)
        # ==============================================================
        if base_domain in KNOWN_LEGITIMATE_DOMAINS:
            if explain:
                return 'LEGITIMATE', 0.99, f"Whitelisted domain: {base_domain}"
            return 'LEGITIMATE', 0.99
        
        # ==============================================================
        # RULE 2: Typo-squatting check (high confidence phishing)
        # ==============================================================
        if check_typosquatting(domain):
            if explain:
                return 'PHISHING', 0.95, f"Typo-squatting detected in domain"
            return 'PHISHING', 0.95
        
        # ==============================================================
        # RULE 3: IP address check (medium confidence phishing)
        # ==============================================================
        if is_ip_address(domain):
            # Local IPs are okay
            if domain.startswith('192.168.') or domain.startswith('10.') or domain.startswith('127.'):
                reasons.append("Local IP address")
            else:
                if explain:
                    return 'PHISHING', 0.80, "Public IP address instead of domain"
                return 'PHISHING', 0.80
        
        # ==============================================================
        # RULE 4: URL shortener (neutral - flag for user awareness)
        # ==============================================================
        if base_domain in URL_SHORTENERS:
            if explain:
                return 'SUSPICIOUS', 0.50, f"URL shortener ({base_domain}) - cannot verify final destination"
            return 'SUSPICIOUS', 0.50
        
        # ==============================================================
        # RULE 5: Suspicious keywords (boost phishing score)
        # ==============================================================
        keyword_count = count_suspicious_keywords(url)
        if keyword_count >= 3:
            if explain:
                return 'PHISHING', 0.85, f"Multiple suspicious keywords ({keyword_count})"
            return 'PHISHING', 0.85
        
        # ==============================================================
        # RULE 6: ML Model (fallback)
        # ==============================================================
        features = extract_url_features(url)
        df = pd.DataFrame([features])
        X_scaled = self.scaler.transform(df)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        label = 'PHISHING' if prediction == 1 else 'LEGITIMATE'
        confidence = probabilities[prediction]
        
        # Adjust confidence based on keyword presence
        if keyword_count >= 2 and label == 'PHISHING':
            confidence = min(0.95, confidence + 0.10)
            reasons.append(f"{keyword_count} suspicious keywords")
        
        if explain:
            reason = f"ML model prediction ({confidence:.1%})"
            if reasons:
                reason += f" + {', '.join(reasons)}"
            return label, confidence, reason
        
        return label, confidence
    
    def predict_batch(self, urls, explain=False):
        """Predict multiple URLs"""
        results = []
        for url in urls:
            try:
                if explain:
                    label, conf, reason = self.predict(url, explain=True)
                    results.append({
                        'url': url,
                        'prediction': label,
                        'confidence': f"{conf:.1%}",
                        'reason': reason
                    })
                else:
                    label, conf = self.predict(url)
                    results.append({
                        'url': url,
                        'prediction': label,
                        'confidence': f"{conf:.1%}"
                    })
            except Exception as e:
                results.append({
                    'url': url,
                    'prediction': 'ERROR',
                    'confidence': str(e)
                })
        return pd.DataFrame(results)


# =================================================================
# EXAMPLE
# =================================================================
if __name__ == "__main__":
    print("="*80)
    print("HYBRID PHISHING DETECTOR (RULES + ML)")
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
        'http://www.paypa1.com',  # Typo-squatting!
        'https://www.microsoft.com',
        'http://free-iphone-winner-claim-now.com',
    ]
    
    print("\n" + "="*80)
    print("TESTING WITH EXPLANATIONS")
    print("="*80)
    
    for i, url in enumerate(test_urls, 1):
        label, conf, reason = detector.predict(url, explain=True)
        
        if label == 'PHISHING':
            status = "🚨"
        elif label == 'SUSPICIOUS':
            status = "⚠️"
        else:
            status = "✓"
        
        print(f"\n[{i}] {url}")
        print(f"    {status} {label} ({conf:.1%})")
        print(f"    Reason: {reason}")
    
    print("\n" + "="*80)
    print("✓ HYBRID APPROACH FIXES OBVIOUS ISSUES!")
    print("="*80)