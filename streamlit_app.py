"""
Phishing URL Detector - Streamlit Web Application
Professional, interactive interface for URL security analysis
"""

import streamlit as st
import re
from urllib.parse import urlparse
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .phishing-alert {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .legitimate-alert {
        background-color: #D1FAE5;
        border-left: 5px solid #059669;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .suspicious-alert {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================
# CONFIGURATION & DATA
# =================================================================

KNOWN_LEGITIMATE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'linkedin.com', 'github.com', 'stackoverflow.com', 'reddit.com', 'wikipedia.org',
    'amazon.com', 'ebay.com', 'walmart.com', 'target.com', 'bestbuy.com',
    'apple.com', 'microsoft.com', 'ibm.com', 'oracle.com', 'adobe.com',
    'netflix.com', 'spotify.com', 'twitch.tv', 'discord.com', 'zoom.us',
    'gmail.com', 'outlook.com', 'yahoo.com', 'protonmail.com',
    'paypal.com', 'stripe.com', 'square.com', 'venmo.com',
    'dropbox.com', 'onedrive.com', 'icloud.com', 'box.com',
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
    return sum(1 for char in text.lower() if char in 'aeiou')

def has_client_server_words(domain):
    keywords = ['client', 'server', 'admin', 'host', 'user']
    return 1 if any(word in domain.lower() for word in keywords) else 0

def get_base_domain(domain):
    parts = domain.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return domain

def check_typosquatting(domain):
    domain_lower = domain.lower()
    for pattern in TYPOSQUATTING_PATTERNS:
        if pattern in domain_lower:
            return True, pattern
    return False, None

def count_suspicious_keywords(url):
    url_lower = url.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
    return len(found_keywords), found_keywords

def is_ip_address(domain):
    return bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$', domain))

def extract_url_features(url):
    """Extract 99 features from URL"""
    features = {}
    parsed = urlparse(url)
    
    domain = parsed.netloc
    directory = parsed.path.rsplit('/', 1)[0] if '/' in parsed.path else ''
    file_part = parsed.path.rsplit('/', 1)[1] if '/' in parsed.path else ''
    params = parsed.query
    
    # URL character counts
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
    
    # Domain features
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
    
    # Directory features
    has_directory = len(directory) > 0
    for feat in ['dot', 'hyphen', 'underline', 'slash', 'questionmark', 'equal', 
                 'at', 'and', 'exclamation', 'space', 'tilde', 'comma', 'plus', 
                 'asterisk', 'hashtag', 'dollar', 'percent']:
        char = {'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
                'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
                'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
                'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
                'percent': '%'}[feat]
        features[f'qty_{feat}_directory'] = float(directory.count(char)) if has_directory else -1.0
    features['directory_length'] = float(len(directory)) if has_directory else -1.0
    
    # File features
    has_file = len(file_part) > 0
    for feat in ['dot', 'hyphen', 'underline', 'slash', 'questionmark', 'equal',
                 'at', 'and', 'exclamation', 'space', 'tilde', 'comma', 'plus',
                 'asterisk', 'hashtag', 'dollar', 'percent']:
        char = {'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
                'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
                'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
                'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
                'percent': '%'}[feat]
        features[f'qty_{feat}_file'] = float(file_part.count(char)) if has_file else -1.0
    features['file_length'] = float(len(file_part)) if has_file else -1.0
    
    # Parameters features
    has_params = len(params) > 0
    for feat in ['dot', 'hyphen', 'underline', 'slash', 'questionmark', 'equal',
                 'at', 'and', 'exclamation', 'space', 'tilde', 'comma', 'plus',
                 'asterisk', 'hashtag', 'dollar', 'percent']:
        char = {'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
                'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
                'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
                'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
                'percent': '%'}[feat]
        features[f'qty_{feat}_params'] = float(params.count(char)) if has_params else -1.0
    features['params_length'] = float(len(params)) if has_params else -1.0
    features['tld_present_params'] = -1.0
    features['qty_params'] = float(len(params.split('&'))) if has_params else -1.0
    
    # Simple features
    features['email_in_url'] = float(1 if '@' in url else 0)
    features['tls_ssl_certificate'] = float(1 if url.startswith('https://') else 0)
    features['url_shortened'] = float(1 if any(s in domain for s in ['bit.ly', 'goo.gl', 'tinyurl.com']) else 0)
    
    return features


@st.cache_resource
def load_model():
    """Load the ML model (cached)"""
    try:
        model = joblib.load('phishing_model_url_only.pkl')
        scaler = joblib.load('scaler_url_only.pkl')
        return model, scaler
    except:
        st.error("⚠️ Model files not found! Please ensure 'phishing_model_url_only.pkl' and 'scaler_url_only.pkl' are in the same directory.")
        return None, None


def analyze_url(url, model, scaler):
    """Main analysis function"""
    parsed = urlparse(url)
    domain = parsed.netloc
    base_domain = get_base_domain(domain)
    
    analysis = {
        'url': url,
        'domain': domain,
        'base_domain': base_domain,
        'prediction': None,
        'confidence': 0,
        'risk_level': 'Unknown',
        'reasons': [],
        'flags': [],
        'suggestions': []
    }
    
    # Rule-based checks
    if base_domain in KNOWN_LEGITIMATE_DOMAINS:
        analysis['prediction'] = 'LEGITIMATE'
        analysis['confidence'] = 0.99
        analysis['risk_level'] = 'Low'
        analysis['reasons'].append(f"✅ Whitelisted domain: {base_domain}")
        return analysis
    
    is_typo, typo_pattern = check_typosquatting(domain)
    if is_typo:
        analysis['prediction'] = 'PHISHING'
        analysis['confidence'] = 0.95
        analysis['risk_level'] = 'Critical'
        analysis['reasons'].append(f"🚨 Typo-squatting detected: '{typo_pattern}'")
        analysis['flags'].append('Typo-squatting')
        return analysis
    
    if is_ip_address(domain):
        if not (domain.startswith('192.168.') or domain.startswith('10.') or domain.startswith('127.')):
            analysis['prediction'] = 'PHISHING'
            analysis['confidence'] = 0.80
            analysis['risk_level'] = 'High'
            analysis['reasons'].append("⚠️ Uses IP address instead of domain name")
            analysis['flags'].append('IP Address')
            return analysis
    
    if base_domain in URL_SHORTENERS:
        analysis['prediction'] = 'SUSPICIOUS'
        analysis['confidence'] = 0.50
        analysis['risk_level'] = 'Medium'
        analysis['reasons'].append(f"⚠️ URL shortener detected: {base_domain}")
        analysis['flags'].append('URL Shortener')
        return analysis
    
    keyword_count, keywords = count_suspicious_keywords(url)
    if keyword_count >= 3:
        analysis['prediction'] = 'PHISHING'
        analysis['confidence'] = 0.85
        analysis['risk_level'] = 'High'
        analysis['reasons'].append(f"🚨 Multiple suspicious keywords found: {', '.join(keywords[:3])}")
        analysis['flags'].extend(keywords[:3])
        return analysis
    
    # ML Model prediction
    features = extract_url_features(url)
    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    label = 'PHISHING' if prediction == 1 else 'LEGITIMATE'
    confidence = probabilities[prediction]
    
    if keyword_count >= 2 and label == 'PHISHING':
        confidence = min(0.95, confidence + 0.10)
        analysis['flags'].extend(keywords)
    
    analysis['prediction'] = label
    analysis['confidence'] = confidence
    analysis['risk_level'] = get_risk_level(label, confidence)
    analysis['reasons'].append(f"🤖 ML Model prediction: {label} ({confidence:.1%} confidence)")
    
    if keyword_count > 0:
        analysis['reasons'].append(f"⚠️ Found {keyword_count} suspicious keyword(s): {', '.join(keywords)}")
    
    # Additional analysis
    if not url.startswith('https://'):
        analysis['flags'].append('No HTTPS')
        analysis['reasons'].append("⚠️ No HTTPS encryption")
    
    if len(domain) > 30:
        analysis['flags'].append('Long domain')
        analysis['reasons'].append(f"⚠️ Unusually long domain name ({len(domain)} characters)")
    
    return analysis


def get_risk_level(prediction, confidence):
    """Determine risk level"""
    if prediction == 'PHISHING':
        if confidence >= 0.90:
            return 'Critical'
        elif confidence >= 0.75:
            return 'High'
        else:
            return 'Medium'
    elif prediction == 'SUSPICIOUS':
        return 'Medium'
    else:
        if confidence >= 0.90:
            return 'Low'
        else:
            return 'Medium'


def get_safety_suggestions(analysis):
    """Generate safety recommendations"""
    suggestions = []
    
    if analysis['prediction'] == 'PHISHING':
        suggestions.extend([
            "🛑 **DO NOT** enter any personal information on this site",
            "🛑 **DO NOT** click any links or download files from this URL",
            "📧 Report this URL to your email provider if received via email",
            "🔒 Close the browser tab immediately",
            "🔍 Verify the legitimacy by contacting the organization directly through official channels"
        ])
    elif analysis['prediction'] == 'SUSPICIOUS':
        suggestions.extend([
            "⚠️ Exercise extreme caution with this URL",
            "🔗 If it's a shortened link, use a URL expander to see the actual destination",
            "✅ Only proceed if you trust the source that shared this link",
            "🔍 Look for reviews or reports about this domain online"
        ])
    else:
        suggestions.extend([
            "✅ The URL appears legitimate, but always stay vigilant",
            "🔒 Ensure the connection is secure (HTTPS) before entering sensitive data",
            "🔍 Double-check the domain spelling for potential typos",
            "💡 Use strong, unique passwords for each website"
        ])
    
    # Additional context-specific suggestions
    if 'No HTTPS' in analysis['flags']:
        suggestions.append("⚠️ Avoid entering payment or personal information without HTTPS")
    
    if 'URL Shortener' in analysis['flags']:
        suggestions.append("🔗 Use a URL preview service before visiting shortened links")
    
    return suggestions


# =================================================================
# STREAMLIT UI
# =================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🛡️ Phishing URL Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered URL Security Analysis | Protect yourself from phishing attacks</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
        st.title("About")
        st.info(
            """
            This tool uses advanced machine learning and rule-based detection to identify 
            phishing URLs with **~90% accuracy**.
            
            **Features:**
            - ✅ Domain reputation checking
            - ✅ Typo-squatting detection
            - ✅ Suspicious pattern analysis
            - ✅ ML-powered classification
            """
        )
        
        st.markdown("---")
        st.subheader("Statistics")
        st.metric("Model Accuracy", "89.6%")
        st.metric("Features Analyzed", "99")
        st.metric("Known Safe Domains", len(KNOWN_LEGITIMATE_DOMAINS))
        
        st.markdown("---")
        st.caption("⚠️ This tool provides guidance but should not be your only security measure. Always verify suspicious URLs through official channels.")
    
    # Main content
    st.markdown("### 🔍 Enter a URL to analyze")
    
    # URL input with examples
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "URL to analyze",
            placeholder="https://example.com",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🔎 Analyze URL", type="primary")
    
    # Example URLs
    with st.expander("📝 Try these example URLs"):
        examples = {
            "Legitimate": "https://www.github.com",
            "Phishing": "http://paypa1-secure-login.com",
            "Suspicious": "https://bit.ly/example"
        }
        
        cols = st.columns(3)
        for idx, (label, url) in enumerate(examples.items()):
            with cols[idx]:
                if st.button(f"{label}\n`{url}`", key=f"example_{idx}"):
                    url_input = url
                    analyze_button = True
    
    # Analysis
    if analyze_button and url_input:
        # Validate URL
        if not url_input.startswith(('http://', 'https://')):
            st.warning("⚠️ Please include http:// or https:// in the URL")
            st.stop()
        
        with st.spinner("🔍 Analyzing URL..."):
            # Simulate processing time for UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            analysis = analyze_url(url_input, model, scaler)
        
        # Results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Prediction card
        prediction = analysis['prediction']
        confidence = analysis['confidence']
        risk_level = analysis['risk_level']
        
        if prediction == 'PHISHING':
            st.markdown(f"""
            <div class="phishing-alert">
                <h2 style="color: #DC2626; margin: 0;">🚨 PHISHING DETECTED</h2>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">This URL is likely malicious!</p>
            </div>
            """, unsafe_allow_html=True)
            result_color = "#DC2626"
        elif prediction == 'SUSPICIOUS':
            st.markdown(f"""
            <div class="suspicious-alert">
                <h2 style="color: #F59E0B; margin: 0;">⚠️ SUSPICIOUS URL</h2>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">Exercise caution with this URL</p>
            </div>
            """, unsafe_allow_html=True)
            result_color = "#F59E0B"
        else:
            st.markdown(f"""
            <div class="legitimate-alert">
                <h2 style="color: #059669; margin: 0;">✅ LIKELY LEGITIMATE</h2>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">This URL appears safe</p>
            </div>
            """, unsafe_allow_html=True)
            result_color = "#059669"
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", prediction)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.metric("Risk Level", risk_level)
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': result_color},
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 75], 'color': "#FEF3C7"},
                    {'range': [75, 100], 'color': "#D1FAE5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Analysis Details")
            for reason in analysis['reasons']:
                st.markdown(f"- {reason}")
            
            if analysis['flags']:
                st.markdown("### 🚩 Flags Detected")
                for flag in set(analysis['flags']):
                    st.markdown(f"- `{flag}`")
        
        with col2:
            st.markdown("### 💡 Safety Recommendations")
            suggestions = get_safety_suggestions(analysis)
            for suggestion in suggestions:
                st.markdown(f"{suggestion}")
        
        # URL breakdown
        with st.expander("🔧 Technical URL Breakdown"):
            parsed = urlparse(url_input)
            st.json({
                "Protocol": parsed.scheme or "N/A",
                "Domain": parsed.netloc or "N/A",
                "Path": parsed.path or "/",
                "Query Parameters": parsed.query or "None",
                "Fragment": parsed.fragment or "None",
                "Uses HTTPS": "Yes" if url_input.startswith('https://') else "No",
                "Domain Length": len(parsed.netloc),
                "Total URL Length": len(url_input)
            })
    
    elif analyze_button:
        st.warning("⚠️ Please enter a URL to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748B;">
            <p>Made with ❤️ using Streamlit | Model Accuracy: 89.6% | Always verify suspicious URLs independently</p>
            <p style="font-size: 0.8rem;">⚠️ This tool is for educational purposes. Do not rely solely on this for critical security decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()