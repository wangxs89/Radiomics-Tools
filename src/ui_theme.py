"""Custom CSS styles for the Radiomics Web App"""

APP_CSS = """
<style>
/* ===== Root Variables ===== */
:root {
    --color-primary: #1E3A5F;
    --color-primary-light: #2A4F7A;
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
}

/* ===== Global ===== */
.block-container {
    max-width: 1280px;
    padding: 2rem 2.5rem 3rem;
}

/* ===== Header ===== */
.main-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2A4F7A 100%);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: var(--shadow-lg);
}
.main-header h1 {
    color: white !important;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: rgba(255,255,255,0.8) !important;
    font-size: 1rem;
    margin: 0;
}

/* ===== Section Cards ===== */
.section-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease;
}
.section-card:hover {
    box-shadow: var(--shadow-md);
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
}
.step-circle.active {
    background: var(--color-primary);
    color: white;
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
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: var(--color-surface);
    border-right: 1px solid var(--color-border);
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
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
