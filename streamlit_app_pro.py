"""
Enhanced Phishing URL Detector - Streamlit Web Application
Professional interface with advanced features:
- Batch URL analysis
- Download reports (PDF/CSV)
- URL history tracking
- Dark mode toggle
- Analytics dashboard
- Screenshot capture (optional)
"""

import streamlit as st
import re
from urllib.parse import urlparse
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import base64
from io import BytesIO
import idna  # For punycode handling

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'phishing_detected' not in st.session_state:
    st.session_state.phishing_detected = 0

# Custom CSS (simplified - removed dark mode styling as Streamlit handles it)
st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #2563EB;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.4rem;
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
        opacity: 0.9;
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
        background: linear-gradient(135deg, #2563EB15 0%, #2563EB05 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #2563EB30;
        text-align: center;
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
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .history-item {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
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

# Common homograph characters used in IDN attacks
HOMOGRAPH_CHARS = {
    'а': 'a',  # Cyrillic a
    'е': 'e',  # Cyrillic e
    'о': 'o',  # Cyrillic o
    'р': 'p',  # Cyrillic p
    'с': 'c',  # Cyrillic c
    'у': 'y',  # Cyrillic y
    'х': 'x',  # Cyrillic x
    'ѕ': 's',  # Cyrillic s
    'і': 'i',  # Cyrillic i
    'ј': 'j',  # Cyrillic j
    'ԁ': 'd',  # Cyrillic d
    'ɡ': 'g',  # Latin small letter g
    'һ': 'h',  # Cyrillic h
    'ӏ': 'l',  # Cyrillic l
    'ո': 'n',  # Armenian n
    'ᴑ': 'o',  # Latin small letter o
    'ԛ': 'q',  # Cyrillic q
    'ѡ': 'w',  # Cyrillic w
}

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

# Helper functions (keeping the same as before)
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

def decode_punycode(domain):
    """
    Decode punycode domain and detect IDN homograph attacks
    Returns: (decoded_domain, is_punycode, has_homographs, homograph_warning)
    """
    is_punycode = False
    has_homographs = False
    homograph_warning = None
    decoded = domain
    
    try:
        # Check if domain contains punycode (xn--)
        if 'xn--' in domain.lower():
            is_punycode = True
            # Decode using idna library
            decoded = idna.decode(domain)
            
            # Check for homograph characters
            homograph_chars_found = []
            for char in decoded:
                if char in HOMOGRAPH_CHARS:
                    has_homographs = True
                    homograph_chars_found.append(f"{char}→{HOMOGRAPH_CHARS[char]}")
            
            if has_homographs:
                # Check if decoded looks like a legitimate domain
                decoded_ascii = ''.join([HOMOGRAPH_CHARS.get(c, c) for c in decoded])
                if decoded_ascii.lower() in KNOWN_LEGITIMATE_DOMAINS:
                    homograph_warning = f"Homograph attack! Looks like '{decoded_ascii}' but contains: {', '.join(homograph_chars_found[:3])}"
                else:
                    homograph_warning = f"Contains suspicious characters: {', '.join(homograph_chars_found[:3])}"
        else:
            # Check for non-ASCII characters even without xn--
            if not all(ord(c) < 128 for c in domain):
                has_homographs = True
                homograph_chars_found = []
                for char in domain:
                    if char in HOMOGRAPH_CHARS:
                        homograph_chars_found.append(f"{char}→{HOMOGRAPH_CHARS[char]}")
                
                if homograph_chars_found:
                    # Try to convert to ASCII equivalent
                    ascii_equiv = ''.join([HOMOGRAPH_CHARS.get(c, c) for c in domain])
                    if ascii_equiv.lower() in KNOWN_LEGITIMATE_DOMAINS:
                        homograph_warning = f"Homograph attack! Impersonates '{ascii_equiv}' using: {', '.join(homograph_chars_found[:3])}"
                    else:
                        homograph_warning = f"Non-ASCII characters detected: {', '.join(homograph_chars_found[:3])}"
    
    except Exception as e:
        # If decoding fails, it might still be suspicious
        pass
    
    return decoded, is_punycode, has_homographs, homograph_warning

def check_full_url_for_phishing(url):
    """
    Check entire URL including query parameters for suspicious domains
    This catches cases like: google.com/search?q=phishing-domain.com
    Returns: (is_suspicious, suspicious_domain, reason)
    """
    try:
        import urllib.parse
        
        # Decode URL-encoded characters
        decoded_url = urllib.parse.unquote(url)
        
        # Extract all potential domains from the entire URL
        # Look for patterns like: .com, .net, .org, etc.
        domain_pattern = r'([a-zA-Z0-9а-яА-Я-]+\.)+[a-zA-Z]{2,}'
        potential_domains = re.findall(domain_pattern, decoded_url)
        
        parsed = urllib.parse.urlparse(url)
        actual_domain = parsed.netloc
        
        # Check each potential domain found in the URL
        for match in potential_domains:
            # Skip if it's the actual domain
            if match in actual_domain:
                continue
            
            # Clean up the match
            domain_candidate = match.strip('.')
            
            # Check for homographs in this domain
            _, _, has_homographs, warning = decode_punycode(domain_candidate)
            
            if has_homographs and warning:
                return True, domain_candidate, f"Suspicious domain in URL: {warning}"
            
            # Check if it looks like a well-known domain (potential phishing)
            base = get_base_domain(domain_candidate)
            
            # Check for typosquatting patterns
            is_typo, pattern = check_typosquatting(domain_candidate)
            if is_typo:
                return True, domain_candidate, f"Typosquatting pattern '{pattern}' found in URL parameters"
            
            # Check if it mimics a legitimate domain
            for legit_domain in KNOWN_LEGITIMATE_DOMAINS:
                # Simple similarity check
                if legit_domain in domain_candidate.lower() or domain_candidate.lower() in legit_domain:
                    if domain_candidate.lower() != actual_domain.lower():
                        return True, domain_candidate, f"Suspicious domain '{domain_candidate}' found in URL (mimics {legit_domain})"
        
        # Check for URL-encoded homograph characters
        if '%D0%' in url or '%D1%' in url:  # Cyrillic characters in URL encoding
            return True, "URL-encoded", "URL contains encoded Cyrillic characters (potential homograph attack)"
        
        # Check for other suspicious Unicode ranges
        if any(encoded in url for encoded in ['%C2%', '%C3%', '%E2%']):
            # Decode and check
            if not all(ord(c) < 128 for c in decoded_url):
                # Check if decoded URL contains homographs
                for char in decoded_url:
                    if char in HOMOGRAPH_CHARS:
                        return True, "URL-encoded", f"URL contains encoded homograph character: {char}→{HOMOGRAPH_CHARS[char]}"
        
    except Exception as e:
        pass
    
    return False, None, None

def convert_to_punycode(domain):
    """
    Convert domain to punycode if it contains non-ASCII characters
    Returns: (punycode_domain, was_converted)
    """
    try:
        # Try to encode to punycode
        punycode = idna.encode(domain).decode('ascii')
        was_converted = (punycode != domain)
        return punycode, was_converted
    except:
        return domain, False

def extract_url_features(url):
    """Extract 99 features from URL with punycode handling"""
    features = {}
    parsed = urlparse(url)
    
    # Decode punycode if present for more accurate feature extraction
    domain = parsed.netloc
    decoded_domain, is_punycode, _, _ = decode_punycode(domain)
    
    # Use decoded domain for feature extraction
    working_domain = decoded_domain if is_punycode else domain
    
    directory = parsed.path.rsplit('/', 1)[0] if '/' in parsed.path else ''
    file_part = parsed.path.rsplit('/', 1)[1] if '/' in parsed.path else ''
    params = parsed.query
    
    # URL character counts (simplified for space)
    for char, name in [('.', 'dot'), ('-', 'hyphen'), ('_', 'underline'), ('/', 'slash'),
                       ('?', 'questionmark'), ('=', 'equal'), ('@', 'at'), ('&', 'and'),
                       ('!', 'exclamation'), (' ', 'space'), ('~', 'tilde'), (',', 'comma'),
                       ('+', 'plus'), ('*', 'asterisk'), ('#', 'hashtag'), ('$', 'dollar'),
                       ('%', 'percent')]:
        features[f'qty_{name}_url'] = float(url.count(char))
    
    features['qty_tld_url'] = float(url.count('.'))
    features['length_url'] = float(len(url))
    
    # Domain features (use decoded domain)
    for char, name in [('.', 'dot'), ('-', 'hyphen'), ('_', 'underline'), ('/', 'slash'),
                       ('?', 'questionmark'), ('=', 'equal'), ('@', 'at'), ('&', 'and'),
                       ('!', 'exclamation'), (' ', 'space'), ('~', 'tilde'), (',', 'comma'),
                       ('+', 'plus'), ('*', 'asterisk'), ('#', 'hashtag'), ('$', 'dollar'),
                       ('%', 'percent')]:
        features[f'qty_{name}_domain'] = float(working_domain.count(char))
    
    features['qty_vowels_domain'] = float(count_vowels(working_domain))
    features['domain_length'] = float(len(working_domain))
    features['domain_in_ip'] = float(1 if is_ip_address(working_domain) else 0)
    features['server_client_domain'] = float(has_client_server_words(working_domain))
    
    # Directory, file, params features
    for component, comp_name in [(directory, 'directory'), (file_part, 'file'), (params, 'params')]:
        has_component = len(component) > 0
        for char, name in [('.', 'dot'), ('-', 'hyphen'), ('_', 'underline'), ('/', 'slash'),
                           ('?', 'questionmark'), ('=', 'equal'), ('@', 'at'), ('&', 'and'),
                           ('!', 'exclamation'), (' ', 'space'), ('~', 'tilde'), (',', 'comma'),
                           ('+', 'plus'), ('*', 'asterisk'), ('#', 'hashtag'), ('$', 'dollar'),
                           ('%', 'percent')]:
            features[f'qty_{name}_{comp_name}'] = float(component.count(char)) if has_component else -1.0
        features[f'{comp_name}_length'] = float(len(component)) if has_component else -1.0
    
    features['tld_present_params'] = -1.0
    features['qty_params'] = float(len(params.split('&'))) if len(params) > 0 else -1.0
    
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
        st.error("⚠️ Model files not found!")
        return None, None

def analyze_url(url, model, scaler):
    """Main analysis function with punycode detection"""
    parsed = urlparse(url)
    domain = parsed.netloc
    base_domain = get_base_domain(domain)
    
    # FIRST: Check entire URL for embedded phishing domains
    is_url_suspicious, suspicious_domain, url_reason = check_full_url_for_phishing(url)
    
    # Punycode detection and conversion
    decoded_domain, is_punycode, has_homographs, homograph_warning = decode_punycode(domain)
    punycode_domain, was_converted = convert_to_punycode(domain)
    
    analysis = {
        'url': url,
        'domain': domain,
        'base_domain': base_domain,
        'decoded_domain': decoded_domain if is_punycode else domain,
        'is_punycode': is_punycode,
        'has_homographs': has_homographs,
        'punycode_warning': homograph_warning,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prediction': None,
        'confidence': 0,
        'risk_level': 'Unknown',
        'reasons': [],
        'flags': [],
        'suggestions': []
    }
    
    # CRITICAL: Check for embedded phishing in URL parameters
    if is_url_suspicious:
        analysis['prediction'] = 'PHISHING'
        analysis['confidence'] = 0.95
        analysis['risk_level'] = 'Critical'
        analysis['reasons'].append(f"🚨 {url_reason}")
        if suspicious_domain:
            analysis['reasons'].append(f"🔍 Found in URL: {suspicious_domain}")
        analysis['flags'].append('Embedded Phishing Domain')
        return analysis
    
    # CRITICAL: Check for homograph attacks in domain
    if has_homographs and homograph_warning:
        analysis['prediction'] = 'PHISHING'
        analysis['confidence'] = 0.98
        analysis['risk_level'] = 'Critical'
        analysis['reasons'].append(f"🚨 {homograph_warning}")
        analysis['flags'].append('Homograph Attack')
        if is_punycode:
            analysis['reasons'].append(f"🔍 Punycode detected: {domain}")
            analysis['reasons'].append(f"📝 Decodes to: {decoded_domain}")
        return analysis
    
    # Rule-based checks (use decoded domain for checking)
    check_domain = decoded_domain if is_punycode else base_domain
    
    # IMPORTANT: Only whitelist if the ACTUAL domain is legitimate
    # AND there are no suspicious patterns in the full URL
    if check_domain in KNOWN_LEGITIMATE_DOMAINS:
        # But warn if punycode was used
        if is_punycode:
            analysis['prediction'] = 'SUSPICIOUS'
            analysis['confidence'] = 0.70
            analysis['risk_level'] = 'Medium'
            analysis['reasons'].append(f"⚠️ Punycode used for known domain")
            analysis['reasons'].append(f"🔍 Original: {domain}")
            analysis['reasons'].append(f"📝 Decodes to: {decoded_domain}")
            analysis['flags'].append('Punycode')
        else:
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
        analysis['reasons'].append(f"🚨 Typo-squatting: '{typo_pattern}'")
        analysis['flags'].append('Typo-squatting')
        return analysis
    
    if is_ip_address(domain):
        if not (domain.startswith('192.168.') or domain.startswith('10.') or domain.startswith('127.')):
            analysis['prediction'] = 'PHISHING'
            analysis['confidence'] = 0.80
            analysis['risk_level'] = 'High'
            analysis['reasons'].append("⚠️ Public IP address")
            analysis['flags'].append('IP Address')
            return analysis
    
    if base_domain in URL_SHORTENERS:
        analysis['prediction'] = 'SUSPICIOUS'
        analysis['confidence'] = 0.50
        analysis['risk_level'] = 'Medium'
        analysis['reasons'].append(f"⚠️ URL shortener: {base_domain}")
        analysis['flags'].append('URL Shortener')
        return analysis
    
    keyword_count, keywords = count_suspicious_keywords(url)
    if keyword_count >= 3:
        analysis['prediction'] = 'PHISHING'
        analysis['confidence'] = 0.85
        analysis['risk_level'] = 'High'
        analysis['reasons'].append(f"🚨 Suspicious keywords: {', '.join(keywords[:3])}")
        analysis['flags'].extend(keywords[:3])
        return analysis
    
    # ML Model
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
    analysis['reasons'].append(f"🤖 ML Model: {label} ({confidence:.1%})")
    
    if keyword_count > 0:
        analysis['reasons'].append(f"⚠️ {keyword_count} suspicious keyword(s): {', '.join(keywords)}")
    
    if not url.startswith('https://'):
        analysis['flags'].append('No HTTPS')
        analysis['reasons'].append("⚠️ No HTTPS encryption")
    
    if len(domain) > 30:
        analysis['flags'].append('Long domain')
        analysis['reasons'].append(f"⚠️ Long domain ({len(domain)} chars)")
    
    return analysis

def get_risk_level(prediction, confidence):
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
            "🛑 DO NOT enter personal information",
            "🛑 DO NOT click links or download files",
            "📧 Report to your email provider",
            "🔒 Close browser tab immediately",
            "🔍 Contact organization through official channels"
        ])
    elif analysis['prediction'] == 'SUSPICIOUS':
        suggestions.extend([
            "⚠️ Exercise extreme caution",
            "🔗 Use URL expander for shortened links",
            "✅ Only proceed if you trust the source",
            "🔍 Search for reviews online"
        ])
    else:
        suggestions.extend([
            "✅ URL appears legitimate",
            "🔒 Verify HTTPS before entering data",
            "🔍 Check domain spelling",
            "💡 Use strong passwords"
        ])
    
    if 'No HTTPS' in analysis['flags']:
        suggestions.append("⚠️ Avoid entering sensitive data without HTTPS")
    
    return suggestions

def create_pdf_report(analysis):
    """Generate PDF report (simplified - returns text)"""
    report = f"""
PHISHING URL ANALYSIS REPORT
Generated: {analysis['timestamp']}
{'='*60}

URL ANALYZED: {analysis['url']}
DOMAIN: {analysis['domain']}

VERDICT: {analysis['prediction']}
CONFIDENCE: {analysis['confidence']:.1%}
RISK LEVEL: {analysis['risk_level']}

ANALYSIS DETAILS:
{chr(10).join('- ' + r for r in analysis['reasons'])}

FLAGS DETECTED:
{chr(10).join('- ' + f for f in analysis['flags']) if analysis['flags'] else 'None'}

RECOMMENDATIONS:
{chr(10).join(get_safety_suggestions(analysis))}

{'='*60}
This report is for informational purposes only.
Always verify suspicious URLs independently.
    """
    return report

def main():
    # Header
    col1, col2 = st.columns([1, 6])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=100)
    
    with col2:
        st.markdown('<p class="main-header">🛡️ Phishing URL Detector</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced AI-Powered URL Security Analysis with Batch Processing & Reports</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=60)
        st.title("Navigation")
        
        page = st.radio("", ["🔍 Single URL Scan", "📊 Batch Analysis", "📈 Analytics Dashboard", "📜 Scan History"])
        
        st.markdown("---")
        st.subheader("Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Scans", st.session_state.total_scans)
        with col2:
            st.metric("Threats Found", st.session_state.phishing_detected)
        
        if st.session_state.total_scans > 0:
            threat_rate = (st.session_state.phishing_detected / st.session_state.total_scans) * 100
            st.metric("Threat Rate", f"{threat_rate:.1f}%")
    
    # Main content based on page selection
    if page == "🔍 Single URL Scan":
        render_single_scan(model, scaler)
    elif page == "📊 Batch Analysis":
        render_batch_analysis(model, scaler)
    elif page == "📈 Analytics Dashboard":
        render_analytics_dashboard()
    elif page == "📜 Scan History":
        render_scan_history()
    
    # Footer
    st.markdown("---")

def render_single_scan(model, scaler):
    """Single URL scan interface"""
    st.markdown("### 🔍 Enter URL to Analyze")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🔎 Analyze", type="primary")
    
    if analyze_button and url_input:
        if not url_input.startswith(('http://', 'https://')):
            st.warning("⚠️ Please include http:// or https://")
            st.stop()
        
        with st.spinner("🔍 Analyzing..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.005)
                progress.progress(i + 1)
            
            analysis = analyze_url(url_input, model, scaler)
            
            # Update stats
            st.session_state.total_scans += 1
            if analysis['prediction'] == 'PHISHING':
                st.session_state.phishing_detected += 1
            
            # Add to history
            st.session_state.scan_history.insert(0, analysis)
            if len(st.session_state.scan_history) > 50:
                st.session_state.scan_history.pop()
        
        # Display results
        display_analysis_results(analysis)

def render_batch_analysis(model, scaler):
    """Batch URL analysis"""
    st.markdown("### 📊 Batch URL Analysis")
    st.info("Upload a CSV file with a column named 'URL' or paste multiple URLs (one per line)")
    
    tab1, tab2 = st.tabs(["📁 Upload CSV", "📝 Paste URLs"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            if 'URL' not in df.columns:
                st.error("❌ CSV must have a column named 'URL'")
            else:
                st.success(f"✅ Loaded {len(df)} URLs")
                
                if st.button("🚀 Analyze All URLs", type="primary"):
                    analyze_batch_urls(df['URL'].tolist(), model, scaler)
    
    with tab2:
        urls_text = st.text_area("Paste URLs (one per line)", height=200)
        
        if st.button("🚀 Analyze URLs", type="primary") and urls_text:
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            analyze_batch_urls(urls, model, scaler)

def analyze_batch_urls(urls, model, scaler):
    """Process batch URLs"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, url in enumerate(urls):
        status_text.text(f"Analyzing {idx+1}/{len(urls)}: {url[:50]}...")
        
        try:
            analysis = analyze_url(url, model, scaler)
            results.append(analysis)
            
            st.session_state.total_scans += 1
            if analysis['prediction'] == 'PHISHING':
                st.session_state.phishing_detected += 1
        except Exception as e:
            results.append({
                'url': url,
                'prediction': 'ERROR',
                'confidence': 0,
                'risk_level': 'Unknown',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        progress_bar.progress((idx + 1) / len(urls))
    
    status_text.text("✅ Analysis complete!")
    
    # Display results table
    df_results = pd.DataFrame([
        {
            'URL': r['url'],
            'Prediction': r['prediction'],
            'Confidence': f"{r['confidence']:.1%}",
            'Risk Level': r['risk_level'],
            'Timestamp': r['timestamp']
        }
        for r in results
    ])
    
    st.dataframe(df_results, use_container_width=True)
    
    # Summary
    col1, col2, col3 = st.columns(3)
    phishing_count = sum(1 for r in results if r['prediction'] == 'PHISHING')
    legit_count = sum(1 for r in results if r['prediction'] == 'LEGITIMATE')
    suspicious_count = sum(1 for r in results if r['prediction'] == 'SUSPICIOUS')
    
    with col1:
        st.metric("🚨 Phishing", phishing_count)
    with col2:
        st.metric("✅ Legitimate", legit_count)
    with col3:
        st.metric("⚠️ Suspicious", suspicious_count)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_results.to_csv(index=False)
        st.download_button(
            "📥 Download CSV Report",
            csv,
            f"phishing_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    with col2:
        # Text report
        text_report = "\n\n".join([create_pdf_report(r) for r in results if r['prediction'] != 'ERROR'])
        st.download_button(
            "📥 Download Text Report",
            text_report,
            f"phishing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )

def render_analytics_dashboard():
    """Analytics dashboard"""
    st.markdown("### 📈 Analytics Dashboard")
    
    if not st.session_state.scan_history:
        st.info("No scan data yet. Analyze some URLs to see analytics!")
        return
    
    # Prepare data
    df = pd.DataFrame(st.session_state.scan_history)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scans", len(df))
    with col2:
        phishing_pct = (df['prediction'] == 'PHISHING').sum() / len(df) * 100
        st.metric("Phishing %", f"{phishing_pct:.1f}%")
    with col3:
        avg_conf = df['confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    with col4:
        high_risk = (df['risk_level'].isin(['High', 'Critical'])).sum()
        st.metric("High Risk URLs", high_risk)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        pred_counts = df['prediction'].value_counts()
        fig = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title="Prediction Distribution",
            color_discrete_map={
                'PHISHING': '#DC2626',
                'LEGITIMATE': '#059669',
                'SUSPICIOUS': '#F59E0B'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_counts = df['risk_level'].value_counts()
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Risk Level Distribution",
            labels={'x': 'Risk Level', 'y': 'Count'},
            color=risk_counts.index,
            color_discrete_map={
                'Critical': '#DC2626',
                'High': '#F59E0B',
                'Medium': '#FBBF24',
                'Low': '#059669'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top domains scanned
    st.markdown("#### 🌐 Most Scanned Domains")
    domain_counts = df['base_domain'].value_counts().head(10)
    st.bar_chart(domain_counts)
    
    # Recent trend
    if len(df) > 10:
        st.markdown("#### 📊 Recent Scan Trend")
        df['scan_number'] = range(len(df), 0, -1)
        fig = px.scatter(
            df.tail(20),
            x='scan_number',
            y='confidence',
            color='prediction',
            title="Confidence Scores (Last 20 Scans)",
            labels={'scan_number': 'Scan Number', 'confidence': 'Confidence'},
            color_discrete_map={
                'PHISHING': '#DC2626',
                'LEGITIMATE': '#059669',
                'SUSPICIOUS': '#F59E0B'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def render_scan_history():
    """Scan history page"""
    st.markdown("### 📜 Scan History")
    
    if not st.session_state.scan_history:
        st.info("No scan history yet!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_pred = st.multiselect(
            "Filter by Prediction",
            options=['PHISHING', 'LEGITIMATE', 'SUSPICIOUS'],
            default=['PHISHING', 'LEGITIMATE', 'SUSPICIOUS']
        )
    
    with col2:
        filter_risk = st.multiselect(
            "Filter by Risk",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High', 'Medium', 'Low']
        )
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.scan_history = []
            st.rerun()
    
    # Display history
    filtered_history = [
        h for h in st.session_state.scan_history
        if h['prediction'] in filter_pred and h['risk_level'] in filter_risk
    ]
    
    st.markdown(f"**Showing {len(filtered_history)} of {len(st.session_state.scan_history)} scans**")
    
    for idx, item in enumerate(filtered_history[:20]):  # Show last 20
        with st.expander(f"{item['timestamp']} - {item['url'][:50]}... - {item['prediction']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**URL:** {item['url']}")
                st.write(f"**Domain:** {item['domain']}")
                st.write(f"**Prediction:** {item['prediction']}")
                st.write(f"**Confidence:** {item['confidence']:.1%}")
                
            with col2:
                st.write(f"**Risk Level:** {item['risk_level']}")
                st.write(f"**Timestamp:** {item['timestamp']}")
            
            if item.get('reasons'):
                st.markdown("**Reasons:**")
                for reason in item['reasons']:
                    st.write(f"- {reason}")

def display_analysis_results(analysis):
    """Display single URL analysis results"""
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Alert box
    prediction = analysis['prediction']
    confidence = analysis['confidence']
    
    if prediction == 'PHISHING':
        st.markdown(f"""
        <div class="phishing-alert">
            <h2 style="color: #DC2626; margin: 0;">🚨 PHISHING DETECTED</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">This URL is likely malicious!</p>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 'SUSPICIOUS':
        st.markdown(f"""
        <div class="suspicious-alert">
            <h2 style="color: #F59E0B; margin: 0;">⚠️ SUSPICIOUS URL</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">Exercise caution</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="legitimate-alert">
            <h2 style="color: #059669; margin: 0;">✅ LIKELY LEGITIMATE</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">URL appears safe</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", prediction)
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with col3:
        st.metric("Risk Level", analysis['risk_level'])
    
    # Gauge
    result_color = {'PHISHING': '#DC2626', 'SUSPICIOUS': '#F59E0B', 'LEGITIMATE': '#059669'}[prediction]
    
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
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Analysis Details")
        for reason in analysis['reasons']:
            st.markdown(f"- {reason}")
        
        if analysis['flags']:
            st.markdown("### 🚩 Flags")
            for flag in set(analysis['flags']):
                st.markdown(f"- `{flag}`")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        for suggestion in get_safety_suggestions(analysis):
            st.markdown(f"{suggestion}")
    
    # Download report
    report_text = create_pdf_report(analysis)
    st.download_button(
        "📥 Download Report",
        report_text,
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        "text/plain"
    )
    
    # Technical details
    with st.expander("🔧 Technical Breakdown"):
        parsed = urlparse(analysis['url'])
        
        tech_info = {
            "Protocol": parsed.scheme or "N/A",
            "Domain": parsed.netloc or "N/A",
            "Path": parsed.path or "/",
            "Query": parsed.query or "None",
            "HTTPS": "Yes" if analysis['url'].startswith('https://') else "No",
            "Domain Length": len(parsed.netloc),
            "URL Length": len(analysis['url'])
        }
        
        # Add punycode info if applicable
        if analysis.get('is_punycode'):
            tech_info["Punycode Detected"] = "Yes"
            tech_info["Decoded Domain"] = analysis.get('decoded_domain', 'N/A')
            tech_info["Original (Punycode)"] = parsed.netloc
        
        if analysis.get('has_homographs'):
            tech_info["Homograph Characters"] = "Detected"
            if analysis.get('punycode_warning'):
                tech_info["Warning"] = analysis['punycode_warning']
        
        st.json(tech_info)

if __name__ == "__main__":
    main()
