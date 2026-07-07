"""Custom CSS styles and SVG assets for the Radiomics Web App"""
from html import escape

# ─────────────────────────────────────────────
#  SVG Logo
# ─────────────────────────────────────────────

# Radiomics logo: CT slice with ROI contours + feature data points
# viewBox 0 0 48 48
_RADIOMICS_LOGO_SVG = '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="48" height="48" fill="none">
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
LOGO_SMALL = _RADIOMICS_LOGO_SVG.replace('width="48" height="48"', 'width="20" height="20"')


# ─────────────────────────────────────────────
#  Lucide-style icon set
# ─────────────────────────────────────────────

_ICON_PATHS = {
    'activity': '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    'bar-chart': '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    'book-open': '<path d="M2 4.5A2.5 2.5 0 0 1 4.5 2H10v18H4.5A2.5 2.5 0 0 0 2 22Z"/><path d="M22 4.5A2.5 2.5 0 0 0 19.5 2H14v18h5.5A2.5 2.5 0 0 1 22 22Z"/>',
    'brain-circuit': '<path d="M12 5a3 3 0 0 0-5.7 1.2A3.5 3.5 0 0 0 4 12a4 4 0 0 0 4 6h1"/><path d="M12 5a3 3 0 0 1 5.7 1.2A3.5 3.5 0 0 1 20 12a4 4 0 0 1-4 6h-1"/><path d="M12 5v14"/><circle cx="8" cy="10" r="1"/><circle cx="16" cy="10" r="1"/><path d="M8 10h4"/><path d="M16 10h-4"/><path d="M12 14h3"/><circle cx="15" cy="14" r="1"/>',
    'check-circle': '<path d="M9 12l2 2 4-5"/><circle cx="12" cy="12" r="10"/>',
    'clipboard-list': '<rect width="8" height="4" x="8" y="2" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M8 11h8"/><path d="M8 16h5"/>',
    'database': '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    'download': '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    'eye': '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    'file-spreadsheet': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h8"/><path d="M10 9v10"/><path d="M14 9v10"/>',
    'filter': '<path d="M22 3H2l8 9v7l4 2v-9Z"/>',
    'folder-open': '<path d="M6 14l1.5-5h13L19 20a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v1"/>',
    'github': '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.1-1.3-.4-2.6-1.3-3.5.1-.3.5-1.7-.1-3.5 0 0-1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.4 1.7 5.4 2 5.4 2c-.6 1.8-.2 3.2-.1 3.5A5.2 5.2 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.4.4-.7 1-.8 1.7-.7.3-2.5.9-3.6-1-.7-1.1-1.9-1.2-1.9-1.2"/><path d="M9 18c-4.5 2-5-2-7-2"/>',
    'grid': '<rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/>',
    'info': '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    'layers': '<path d="M12 2 2 7l10 5 10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    'line-chart': '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    'mail': '<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-10 6L2 7"/>',
    'microscope': '<path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 0 0 7-7h-3a4 4 0 0 1-4 4Z"/><path d="M9 14 5 10l7-7 4 4Z"/><path d="m10 12 4 4"/>',
    'network': '<rect width="6" height="6" x="16" y="16" rx="1"/><rect width="6" height="6" x="2" y="16" rx="1"/><rect width="6" height="6" x="9" y="2" rx="1"/><path d="M12 8v4"/><path d="m12 12-7 4"/><path d="m12 12 7 4"/>',
    'save': '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/>',
    'scan': '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
    'settings': '<path d="M12.2 2h-.4l-1 3a7.6 7.6 0 0 0-1.8.8L6.1 4.6l-.3.3L4.6 6.1 5.8 9a7.6 7.6 0 0 0-.8 1.8l-3 1v.4l3 1c.2.6.4 1.2.8 1.8l-1.2 2.9.3.3 1.2 1.2L9 18.2c.6.4 1.2.6 1.8.8l1 3h.4l1-3c.6-.2 1.2-.4 1.8-.8l2.9 1.2.3-.3 1.2-1.2-1.2-2.9c.4-.6.6-1.2.8-1.8l3-1v-.4l-3-1a7.6 7.6 0 0 0-.8-1.8l1.2-2.9-.3-.3-1.2-1.2L15 5.8a7.6 7.6 0 0 0-1.8-.8Z"/><circle cx="12" cy="12" r="3"/>',
    'sliders': '<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><path d="M2 14h4"/><path d="M10 8h4"/><path d="M18 16h4"/>',
    'target': '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    'upload': '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
}


def get_icon_svg(name: str, size: int = 18, stroke_width: float = 2.0) -> str:
    """Return an inline Lucide-style SVG icon."""
    paths = _ICON_PATHS.get(name, _ICON_PATHS['info'])
    return (
        f'<svg class="rt-icon" xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{paths}</svg>'
    )


def get_icon_heading_html(title: str, icon: str, level: int = 3,
                          subtitle: str = "", accent: str = "primary") -> str:
    """Generate a compact heading with a reusable icon badge."""
    safe_title = escape(title)
    safe_subtitle = escape(subtitle)
    level = min(max(level, 2), 5)
    subtitle_html = f'<span class="icon-heading-subtitle">{safe_subtitle}</span>' if safe_subtitle else ''
    return f'''
    <h{level} class="icon-heading icon-heading-{accent}">
        <span class="icon-badge">{get_icon_svg(icon, 18)}</span>
        <span class="icon-heading-copy">
            <span class="icon-heading-title">{safe_title}</span>
            {subtitle_html}
        </span>
    </h{level}>
    '''


def get_sidebar_section_html(title: str, icon: str) -> str:
    """Generate a small icon title for sidebar sections."""
    return f'''
    <p class="sidebar-section-title">
        {get_icon_svg(icon, 15)}
        <span>{escape(title)}</span>
    </p>
    '''


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

/* ===== Icon System ===== */
.rt-icon {
    display: inline-block;
    flex-shrink: 0;
    vertical-align: middle;
}
.icon-heading {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0 0 0.9rem 0;
}
.icon-heading .icon-badge {
    width: 34px;
    height: 34px;
    border-radius: var(--radius-sm);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--color-primary);
    background: rgba(30,58,95,0.08);
    border: 1px solid rgba(30,58,95,0.12);
}
.icon-heading-accent .icon-badge {
    color: var(--color-accent);
    background: rgba(5,150,105,0.08);
    border-color: rgba(5,150,105,0.14);
}
.icon-heading-secondary .icon-badge {
    color: var(--color-secondary);
    background: rgba(37,99,235,0.08);
    border-color: rgba(37,99,235,0.14);
}
.icon-heading h2,
.icon-heading h3,
.icon-heading h4,
.icon-heading h5 {
    margin: 0 !important;
    color: var(--color-text) !important;
    letter-spacing: 0;
    line-height: 1.2;
}
.icon-heading-copy {
    display: inline-flex;
    flex-direction: column;
    min-width: 0;
}
.icon-heading-title {
    display: inline-block;
    margin: 0 !important;
    color: var(--color-text) !important;
    letter-spacing: 0;
    line-height: 1.2;
    font-weight: 700;
}
h2.icon-heading .icon-heading-title {
    font-size: 1.45rem !important;
}
h3.icon-heading .icon-heading-title {
    font-size: 1.22rem !important;
}
h4.icon-heading .icon-heading-title {
    font-size: 1.02rem !important;
}
h5.icon-heading .icon-heading-title {
    font-size: 0.95rem !important;
}
.icon-heading-subtitle {
    display: block;
    margin-top: 0.15rem;
    color: var(--color-text-secondary);
    font-size: 0.86rem;
    line-height: 1.35;
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
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--color-surface);
    border-color: var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
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
.step-circle .rt-icon {
    width: 16px;
    height: 16px;
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
.sidebar-section-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--color-primary);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0;
    margin: 1rem 0 0.55rem;
}
.sidebar-section-title .rt-icon {
    color: var(--color-accent);
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
    step_icons = {
        'Select Data': 'folder-open',
        'Data': 'folder-open',
        'Verify ROI': 'eye',
        'Verify': 'eye',
        'Extract': 'activity',
        'Preprocess': 'sliders',
        'Filters': 'filter',
        'Features': 'grid',
    }
    parts = ['<div class="step-indicator">']
    for i, label in enumerate(steps):
        if i < current:
            circle_cls = "done"
            label_cls = ""
            icon = "&#10003;"
        elif i == current:
            circle_cls = "active"
            label_cls = "active"
            icon = get_icon_svg(step_icons.get(label, 'target'), 16, 2.2)
        else:
            circle_cls = "pending"
            label_cls = ""
            icon = get_icon_svg(step_icons.get(label, 'target'), 16, 2.2)

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
