"""Custom CSS styles and SVG assets for the Radiomics Web App"""

# ─────────────────────────────────────────────
#  SVG Logo
# ─────────────────────────────────────────────

# Radiomics logo: CT slice with ROI contours + feature data points
# viewBox 0 0 48 48
_RADIOMICS_LOGO_SVG = '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none">
  <!-- Outer ring: CT slice boundary -->
  <circle cx="24" cy="24" r="21" stroke="white" stroke-width="1.5" opacity="0.9"/>
  <circle cx="24" cy="24" r="18" stroke="white" stroke-width="0.5" opacity="0.3"/>
  <!-- ROI contour: irregular tumor shape -->
  <path d="M20 14 C24 12, 30 14, 32 18 C34 22, 32 28, 28 30 C24 32, 18 30, 16 26 C14 22, 16 16, 20 14Z"
        stroke="white" stroke-width="1.5" fill="none" opacity="0.95"/>
  <!-- Inner contour -->
  <path d="M22 18 C25 17, 28 18, 29 21 C30 24, 28 27, 25 28 C22 29, 19 27, 18 24 C17 21, 19 19, 22 18Z"
        stroke="white" stroke-width="1" fill="rgba(255,255,255,0.08)" opacity="0.7"/>
  <!-- Feature extraction points -->
  <circle cx="20" cy="16" r="1.5" fill="#10B981" opacity="0.9"/>
  <circle cx="30" cy="20" r="1.5" fill="#10B981" opacity="0.9"/>
  <circle cx="27" cy="28" r="1.5" fill="#10B981" opacity="0.9"/>
  <circle cx="18" cy="24" r="1.5" fill="#10B981" opacity="0.9"/>
  <circle cx="24" cy="22" r="2" fill="white" opacity="0.95"/>
  <!-- Data connection lines -->
  <line x1="20" y1="16" x2="24" y2="22" stroke="#10B981" stroke-width="0.7" opacity="0.5"/>
  <line x1="30" y1="20" x2="24" y2="22" stroke="#10B981" stroke-width="0.7" opacity="0.5"/>
  <line x1="27" y1="28" x2="24" y2="22" stroke="#10B981" stroke-width="0.7" opacity="0.5"/>
  <line x1="18" y1="24" x2="24" y2="22" stroke="#10B981" stroke-width="0.7" opacity="0.5"/>
  <!-- Radiating analysis lines -->
  <line x1="24" y1="3" x2="24" y2="7" stroke="white" stroke-width="0.5" opacity="0.3"/>
  <line x1="45" y1="24" x2="41" y2="24" stroke="white" stroke-width="0.5" opacity="0.3"/>
  <line x1="24" y1="45" x2="24" y2="41" stroke="white" stroke-width="0.5" opacity="0.3"/>
  <line x1="3" y1="24" x2="7" y2="24" stroke="white" stroke-width="0.5" opacity="0.3"/>
</svg>'''

LOGO_LARGE = _RADIOMICS_LOGO_SVG  # 48×48 — header
LOGO_SMALL = _RADIOMICS_LOGO_SVG.replace('viewBox="0 0 48 48"', 'viewBox="0 0 48 48" width="20" height="20"')


# ─────────────────────────────────────────────
#  Decorative SVG
# ─────────────────────────────────────────────

# Hero banner: stylized CT slice with ROI + feature vectors
HERO_BANNER_SVG = '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 80" fill="none" style="width:100%;height:80px;display:block;">
  <!-- Background gradient -->
  <defs>
    <linearGradient id="heroBg" x1="0" y1="0" x2="800" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#1E3A5F" stop-opacity="0.04"/>
      <stop offset="50%" stop-color="#2563EB" stop-opacity="0.02"/>
      <stop offset="100%" stop-color="#059669" stop-opacity="0.04"/>
    </linearGradient>
    <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1E3A5F" stop-opacity="0"/>
      <stop offset="50%" stop-color="#1E3A5F" stop-opacity="0.15"/>
      <stop offset="100%" stop-color="#1E3A5F" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect width="800" height="80" fill="url(#heroBg)"/>
  <!-- Decorative grid lines -->
  <line x1="0" y1="40" x2="800" y2="40" stroke="url(#lineGrad)" stroke-width="0.5"/>
  <!-- Left: CT scan slice -->
  <g transform="translate(120,40)" opacity="0.12">
    <circle r="28" stroke="#1E3A5F" stroke-width="1.5"/>
    <circle r="22" stroke="#1E3A5F" stroke-width="0.5"/>
    <path d="M-6,-10 C4,-14, 14,-8, 16,0 C18,8, 10,16, 2,18 C-6,20, -14,14, -16,6 C-18,-2, -12,-8, -6,-10Z"
          stroke="#1E3A5F" stroke-width="1.2" fill="none"/>
  </g>
  <!-- Center: ROI contour with features -->
  <g transform="translate(400,40)" opacity="0.1">
    <path d="M-20,-14 C-8,-18, 12,-14, 18,-4 C24,6, 18,18, 6,22 C-6,26, -18,18, -22,8 C-26,-2, -28,-10, -20,-14Z"
          stroke="#2563EB" stroke-width="1.5" fill="none"/>
    <circle cx="-10" cy="-8" r="2" fill="#059669"/>
    <circle cx="12" cy="-4" r="2" fill="#059669"/>
    <circle cx="8" cy="12" r="2" fill="#059669"/>
    <circle cx="-12" cy="6" r="2" fill="#059669"/>
    <circle cx="0" cy="0" r="3" fill="#1E3A5F"/>
    <line x1="-10" y1="-8" x2="0" y2="0" stroke="#059669" stroke-width="0.8"/>
    <line x1="12" y1="-4" x2="0" y2="0" stroke="#059669" stroke-width="0.8"/>
    <line x1="8" y1="12" x2="0" y2="0" stroke="#059669" stroke-width="0.8"/>
    <line x1="-12" y1="6" x2="0" y2="0" stroke="#059669" stroke-width="0.8"/>
  </g>
  <!-- Right: data chart motif -->
  <g transform="translate(680,40)" opacity="0.1">
    <line x1="-20" y1="16" x2="-20" y2="-16" stroke="#1E3A5F" stroke-width="1"/>
    <line x1="-20" y1="16" x2="20" y2="16" stroke="#1E3A5F" stroke-width="1"/>
    <polyline points="-15,8 -8,-4 0,2 8,-10 15,-6" stroke="#2563EB" stroke-width="1.5" fill="none"/>
    <circle cx="-15" cy="8" r="1.5" fill="#2563EB"/>
    <circle cx="-8" cy="-4" r="1.5" fill="#2563EB"/>
    <circle cx="0" cy="2" r="1.5" fill="#2563EB"/>
    <circle cx="8" cy="-10" r="1.5" fill="#2563EB"/>
    <circle cx="15" cy="-6" r="1.5" fill="#2563EB"/>
  </g>
  <!-- Scattered dots across banner -->
  <circle cx="50" cy="20" r="1" fill="#1E3A5F" opacity="0.06"/>
  <circle cx="200" cy="60" r="1.5" fill="#2563EB" opacity="0.06"/>
  <circle cx="350" cy="15" r="1" fill="#059669" opacity="0.06"/>
  <circle cx="550" cy="65" r="1.5" fill="#1E3A5F" opacity="0.06"/>
  <circle cx="750" cy="25" r="1" fill="#2563EB" opacity="0.06"/>
  <circle cx="280" cy="70" r="0.8" fill="#059669" opacity="0.05"/>
  <circle cx="620" cy="12" r="1.2" fill="#1E3A5F" opacity="0.05"/>
</svg>'''


# ─────────────────────────────────────────────
#  CSS Styles
# ─────────────────────────────────────────────

APP_CSS = """
<style>
/* ===== Root Variables ===== */
:root {
    --color-primary: #1E3A5F;
    --color-primary-light: #2A4F7A;
    --color-primary-dark: #152B47;
    --color-secondary: #2563EB;
    --color-accent: #059669;
    --color-accent-light: #10B981;
    --color-bg: #F8FAFC;
    --color-surface: #FFFFFF;
    --color-text: #0F172A;
    --color-text-secondary: #475569;
    --color-text-muted: #94A3B8;
    --color-border: #E2E8F0;
    --color-error: #DC2626;
    --color-warning: #D97706;
    --color-success: #059669;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
    --shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.04);
}

/* ===== Page Background ===== */
body {
    background-color: var(--color-bg);
    background-image:
        radial-gradient(circle at 1px 1px, rgba(30,58,95,0.04) 1px, transparent 0);
    background-size: 28px 28px;
}

/* ===== Global ===== */
.block-container {
    max-width: 1280px;
    padding: 2rem 2.5rem 3rem;
}

/* ===== Header ===== */
.main-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2A4F7A 50%, #1E3A5F 100%);
    border-radius: var(--radius-lg);
    padding: 2.25rem 2.5rem;
    margin-bottom: 0.5rem;
    color: white;
    box-shadow: var(--shadow-xl);
    position: relative;
    overflow: hidden;
}
/* Subtle concentric circle decoration in header */
.main-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.06);
    pointer-events: none;
}
.main-header::after {
    content: '';
    position: absolute;
    top: -20%;
    right: -5%;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.04);
    pointer-events: none;
}
.main-header-inner {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    position: relative;
    z-index: 1;
}
.main-header-logo {
    flex-shrink: 0;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));
}
.main-header-text h1 {
    color: white !important;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.main-header-text p {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}

/* ===== Hero Banner ===== */
.hero-banner {
    margin-bottom: 1.5rem;
    border-radius: var(--radius-md);
    overflow: hidden;
}

/* ===== Section Cards ===== */
.section-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-md);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.section-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

/* ===== Step Indicator ===== */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2rem;
    padding: 0 0.5rem;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.step-circle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.875rem;
    flex-shrink: 0;
    transition: all 0.2s ease;
}
.step-circle.active {
    background: var(--color-primary);
    color: white;
    box-shadow: 0 0 0 4px rgba(30,58,95,0.15);
}
.step-circle.done {
    background: var(--color-accent);
    color: white;
}
.step-circle.pending {
    background: var(--color-border);
    color: var(--color-text-muted);
}
.step-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--color-text-secondary);
    white-space: nowrap;
}
.step-label.active {
    color: var(--color-primary);
    font-weight: 600;
}
.step-line {
    flex: 1;
    height: 2px;
    background: var(--color-border);
    margin: 0 0.5rem;
    min-width: 20px;
    transition: background 0.2s ease;
}
.step-line.done {
    background: var(--color-accent);
}

/* ===== Buttons ===== */
.stButton > button {
    border-radius: var(--radius-sm);
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.5rem 1.25rem;
    transition: all 0.15s ease;
    border: 1px solid var(--color-border);
}
.stButton > button:hover {
    box-shadow: var(--shadow-sm);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: white;
}
.stButton > button[kind="primary"]:hover {
    background: var(--color-primary-light);
    border-color: var(--color-primary-light);
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
    border-right: 1px solid var(--color-border);
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--color-border);
}
.sidebar-logo-text {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--color-primary);
    letter-spacing: -0.01em;
}

/* ===== Form Elements ===== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
    font-size: 0.9rem;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--color-primary);
    box-shadow: 0 0 0 2px rgba(30,58,95,0.15);
}

/* ===== Multiselect ===== */
.stMultiSelect > div > div {
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
}

/* ===== Slider ===== */
.stSlider > div > div > div > div {
    background: var(--color-primary);
}

/* ===== Status Messages ===== */
[data-testid="stAlertContainer"] > div {
    border-radius: var(--radius-sm);
    border-left: 4px solid;
}

/* ===== Data Display ===== */
.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-primary);
    font-variant-numeric: tabular-nums;
}
.stat-label {
    font-size: 0.8rem;
    color: var(--color-text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ===== Divider ===== */
hr {
    border: none;
    border-top: 1px solid var(--color-border);
    margin: 1.5rem 0;
}

/* ===== Table ===== */
.dataframe {
    border-radius: var(--radius-sm);
    overflow: hidden;
    font-size: 0.85rem;
}
.dataframe th {
    background: var(--color-bg);
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--color-text-secondary);
}

/* ===== Expander ===== */
.streamlit-expanderHeader {
    font-weight: 500;
    color: var(--color-text-secondary);
}

/* ===== Footer ===== */
.app-footer {
    text-align: center;
    color: var(--color-text-muted);
    font-size: 0.78rem;
    padding: 1.5rem 0 1rem;
    border-top: 1px solid var(--color-border);
    margin-top: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.app-footer a {
    color: var(--color-text-muted);
    text-decoration: none;
    transition: color 0.15s ease;
}
.app-footer a:hover {
    color: var(--color-primary);
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--color-text-muted);
}

/* ===== Reduced Motion ===== */
@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
        animation: none !important;
    }
}
</style>
"""


def get_step_indicator_html(steps: list, current: int) -> str:
    """Generate step indicator HTML.

    Args:
        steps: list of step label strings
        current: 0-based index of the current step
    """
    parts = ['<div class="step-indicator">']
    for i, label in enumerate(steps):
        if i < current:
            circle_cls = "done"
            label_cls = ""
            icon = "&#10003;"
        elif i == current:
            circle_cls = "active"
            label_cls = "active"
            icon = str(i + 1)
        else:
            circle_cls = "pending"
            label_cls = ""
            icon = str(i + 1)

        parts.append(f'''
        <div class="step-item">
            <div class="step-circle {circle_cls}">{icon}</div>
            <span class="step-label {label_cls}">{label}</span>
        </div>''')
        if i < len(steps) - 1:
            line_cls = "done" if i < current else ""
            parts.append(f'<div class="step-line {line_cls}"></div>')

    parts.append('</div>')
    return ''.join(parts)


def get_header_html() -> str:
    """Generate the enhanced header HTML with logo."""
    return f'''
    <div class="main-header">
        <div class="main-header-inner">
            <div class="main-header-logo">{LOGO_LARGE}</div>
            <div class="main-header-text">
                <h1>Radiomics Tool</h1>
                <p>Medical Image Feature Extraction &amp; Analysis Platform</p>
            </div>
        </div>
    </div>
    '''


def get_sidebar_logo_html() -> str:
    """Generate sidebar logo HTML."""
    return f'''
    <div class="sidebar-logo">
        {LOGO_SMALL}
        <span class="sidebar-logo-text">Radiomics Tool</span>
    </div>
    '''


def get_hero_banner_html() -> str:
    """Generate hero banner SVG."""
    return f'<div class="hero-banner">{HERO_BANNER_SVG}</div>'


def get_footer_html() -> str:
    """Generate footer HTML with logo."""
    return '''
    <div class="app-footer">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 48 48" fill="none" opacity="0.4">
            <circle cx="24" cy="24" r="21" stroke="#1E3A5F" stroke-width="1.5"/>
            <path d="M20 14 C24 12, 30 14, 32 18 C34 22, 32 28, 28 30 C24 32, 18 30, 16 26 C14 22, 16 16, 20 14Z"
                  stroke="#1E3A5F" stroke-width="1.5" fill="none"/>
            <circle cx="24" cy="22" r="2" fill="#059669"/>
        </svg>
        <span>Radiomics Tool &copy; 2026</span>
        <span>&middot;</span>
        <span>wangxiaoshen0408@126.com</span>
        <span>&middot;</span>
        <a href="https://github.com/wangxs89/Radiomics-Tools" target="_blank">GitHub</a>
    </div>
    '''
