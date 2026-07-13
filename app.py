import os
import pickle
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from disease_risk import render_disease_risk_module

# 1. Page Configuration (Keep at top, remove emojis)
st.set_page_config(
    page_title="Enterprise Genomics Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Background Override */
    html, body, p, li, label, .stMarkdown, .stText, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        color: #F8FAFC !important;
    }
    
    h1, h2, h3, h4, h5, h6, .panel-subheader, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
    
    .stApp {
        background-color: #020617 !important;
        color: #F8FAFC !important;
    }
    
    /* Lock sidebar open and hide the toggle button to prevent "keyboard_double" text glitches */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
        transform: none !important;
        visibility: visible !important;
    }
    
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }
    
    /* Remove default Streamlit top padding, decoration bars, and white overlays */
    div.block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    #MainMenu {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent !important;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    footer {visibility: hidden;}
    .stMainMenu {visibility: hidden;}
    #stDecoration {display:none;}
    
    /* Custom Sidebar title */
    .sidebar-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        margin-bottom: 2px !important;
    }
    
    .sidebar-subtitle {
        font-size: 0.72rem !important;
        color: #64748B !important;
        margin-bottom: 16px !important;
    }
    
    /* Sliders in Sidebar */
    .stSlider label, .stLabel, .stNumberInput label {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.02em !important;
    }
    
    div[data-baseweb="slider"] {
        padding-bottom: 8px !important;
    }
    div[data-baseweb="slider"] > div {
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    div[data-baseweb="slider"] div[role="progressbar"] {
        background-color: #2563EB !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #06B6D4 !important;
        border: 2px solid #0F172A !important;
        box-shadow: 0 0 8px rgba(6, 182, 212, 0.5) !important;
    }
    
    /* Styling section breaks in sidebar */
    .sidebar-section-header {
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        color: #06B6D4 !important;
        letter-spacing: 0.08em !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding-bottom: 4px !important;
    }

    /* Core Glassmorphism Card System */
    .glass-card {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
        min-height: 145px !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .glass-card:hover {
        border-color: rgba(37, 99, 235, 0.4) !important;
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.12) !important;
        transform: translateY(-2px) !important;
    }

    .kpi-title {
        font-size: 0.68rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }

    .kpi-val {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    .kpi-unit {
        font-size: 0.8rem !important;
        color: #64748B !important;
        margin-left: 4px !important;
        font-weight: 500 !important;
    }
    
    .kpi-percentile {
        font-size: 0.72rem !important;
        color: #10B981 !important;
        font-weight: 500 !important;
        margin-top: 4px !important;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .trend-icon-up {
        font-size: 0.65rem;
    }
    
    .kpi-percentile-desc {
        color: #64748B !important;
        font-weight: 400 !important;
    }
    
    .gauge-center-text {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em !important;
    }

    /* Professional Status Badges */
    .status-badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 4px 10px !important;
        border-radius: 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
    }

    .status-badge.active {
        color: #10B981 !important;
        border-color: rgba(16, 185, 129, 0.25) !important;
    }

    .status-badge.enabled {
        color: #06B6D4 !important;
        border-color: rgba(6, 182, 212, 0.25) !important;
    }

    .status-badge.demo {
        color: #2563EB !important;
        border-color: rgba(37, 99, 235, 0.25) !important;
    }
    
    .indicator-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .indicator-dot.active {
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-green 2s infinite;
    }
    .indicator-dot.enabled {
        background-color: #06B6D4;
        box-shadow: 0 0 8px #06B6D4;
        animation: pulse-cyan 2s infinite;
    }
    .indicator-dot.demo {
        background-color: #2563EB;
        box-shadow: 0 0 8px #2563EB;
        animation: pulse-blue 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { transform: scale(0.9); opacity: 0.6; }
        50% { transform: scale(1.15); opacity: 1; }
    }
    @keyframes pulse-cyan {
        0%, 100% { transform: scale(0.9); opacity: 0.6; }
        50% { transform: scale(1.15); opacity: 1; }
    }
    @keyframes pulse-blue {
        0%, 100% { transform: scale(0.9); opacity: 0.6; }
        50% { transform: scale(1.15); opacity: 1; }
    }

    /* Structured Recommendations */
    .rec-box {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 6px !important;
        padding: 12px 14px !important;
        margin-bottom: 12px !important;
        min-height: 90px;
        transition: all 0.2s ease;
    }
    
    .rec-box:hover {
        border-color: rgba(255, 255, 255, 0.12) !important;
        background: rgba(15, 23, 42, 0.65) !important;
    }
    
    .rec-box.border-elite { border-left: 4px solid #10B981 !important; }
    .rec-box.border-suitable { border-left: 4px solid #06B6D4 !important; }
    .rec-box.border-moderate { border-left: 4px solid #2563EB !important; }
    .rec-box.border-attention { border-left: 4px solid #EF4444 !important; }

    .rec-header {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        color: #E2E8F0 !important;
        font-size: 0.78rem !important;
        margin-bottom: 4px !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }
    
    .rec-body {
        color: #94A3B8 !important;
        font-size: 0.74rem !important;
        line-height: 1.45 !important;
    }

    .rec-status-pill {
        font-size: 0.62rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        padding: 2px 6px !important;
        border-radius: 3px !important;
    }
    
    .pill-elite { background-color: rgba(16, 185, 129, 0.1) !important; color: #10B981 !important; }
    .pill-suitable { background-color: rgba(6, 182, 212, 0.1) !important; color: #06B6D4 !important; }
    .pill-moderate { background-color: rgba(37, 99, 235, 0.1) !important; color: #2563EB !important; }
    .pill-attention { background-color: rgba(239, 68, 68, 0.1) !important; color: #EF4444 !important; }

    /* Section Subheaders */
    .panel-subheader {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #06B6D4 !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
    }
    
    /* Table Styling for Cohorts */
    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.76rem;
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    .benchmark-table th {
        text-align: left;
        color: #94A3B8;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 9px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.8);
    }
    .benchmark-table td {
        padding: 9px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: #E2E8F0;
    }
    .benchmark-table tr:last-child td {
        border-bottom: none;
    }
    .benchmark-table tr:hover {
        background-color: rgba(255, 255, 255, 0.02);
    }
    .text-highlight-blue { color: #2563EB !important; font-weight: 600; }
    .text-highlight-cyan { color: #06B6D4 !important; font-weight: 600; }
    .text-highlight-green { color: #10B981 !important; font-weight: 600; }
    
    /* Quick Insight block styling */
    .insight-block {
        border-left: 3px solid #06B6D4;
        background: rgba(6, 182, 212, 0.04);
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 0.78rem;
        color: #E2E8F0;
        line-height: 1.45;
        margin-bottom: 14px;
    }
    
    /* Override Streamlit Tabs styling to look premium */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent !important;
        border: none !important;
        color: #94A3B8 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0 5px !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #06B6D4 !important;
        border-bottom: 2px solid #06B6D4 !important;
    }
    
    /* Radio Selector override */
    div[data-testid="stRadio"] > label {
        font-size: 0.78rem !important;
        color: #94A3B8 !important;
        margin-bottom: 6px !important;
    }
    div[data-testid="stRadio"] label {
        font-size: 0.74rem !important;
        color: #F8FAFC !important;
    }
    
    /* Styled action button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%) !important;
        border: none !important;
        color: #F8FAFC !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        padding: 9px 24px !important;
        border-radius: 4px !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.15) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3) !important;
        background: linear-gradient(135deg, #1D4ED8 0%, #0891B2 100%) !important;
    }
    
    .stButton>button {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 4px !important;
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.04) !important;
        color: #FFFFFF !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. Model Resources Loading Helper
# ----------------------------------------------------
@st.cache_resource
def load_analysis_resources():
    utils = {}
    try:
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            utils['scaler'] = pickle.load(f)
        
        # Load RF models
        with open('models/model_milk.pkl', 'rb') as f:
            utils['model_milk'] = pickle.load(f)
        with open('models/model_fat.pkl', 'rb') as f:
            utils['model_fat'] = pickle.load(f)
        with open('models/model_protein.pkl', 'rb') as f:
            utils['model_protein'] = pickle.load(f)
        
        # Load metrics
        with open('models/metrics.pkl', 'rb') as f:
            utils['metrics'] = pickle.load(f)
            
        # Load cohort data
        utils['df_m'] = pd.read_csv('data/HO_M.csv').ffill().bfill()
        utils['df_f'] = pd.read_csv('data/HO_f.csv').ffill().bfill()
        utils['df_p'] = pd.read_csv('data/HO_p.csv').ffill().bfill()
        
        # Keep simulated dataframe for percentile mapping
        utils['df_sim'] = pd.read_csv('data/simulated_cattle_data.csv').ffill().bfill()
        
        utils['loaded'] = True
    except FileNotFoundError:
        utils['loaded'] = False
        st.error("Missing resources files in models/ and data/ directories. Run train_model.py first.")
    except Exception as e:
        utils['loaded'] = False
        st.error(f"Error loading system resources: {e}")
    return utils

res = load_analysis_resources()

# ----------------------------------------------------
# 4. Decisional Recommendations Engine
# ----------------------------------------------------
def get_breeding_recommendations(ptam, ptaf, ptap, ptapl, ptadpr, ptascs, gps, pred_milk):
    actions = {}
    
    # 1. Selective Breeding Recommendation
    if gps >= 70.0 and ptadpr >= -1.0 and ptascs <= 3.0:
        actions['breeding'] = "Retain for elite breeding programs. Transmittable health and lactation values indicate the subject will advance physical resilience and milk quality in progeny."
        actions['breeding_pill'] = "Elite Candidate"
        actions['breeding_class'] = "elite"
    elif gps >= 48.0 and ptadpr >= -2.0 and ptascs <= 3.3:
        actions['breeding'] = "Retain for commercial lines. Stabilize daughter conception and cell counts of progeny by mating exclusively with premium high-index sires."
        actions['breeding_pill'] = "Suitable"
        actions['breeding_class'] = "suitable"
    else:
        actions['breeding'] = "Negative breeding outlook. Below average genetic potential. Subject recommended for industrial milk output; cull replacements from this lineage."
        actions['breeding_pill'] = "Commercial Only"
        actions['breeding_class'] = "attention"
        
    # 2. Milk Yield Strategy
    if ptam > 400:
        actions['milk'] = "Superior milk potential detected. Transmitted lactation volume projection in top cohort. Focus on additive rations to support maximum metabolic output."
        actions['milk_pill'] = "High Yield"
        actions['milk_class'] = "elite"
    elif ptam > -300:
        actions['milk'] = "Average breed lactation potential. Standard replacement performance. Subject will execute stable commercial output under baseline nutrition."
        actions['milk_pill'] = "Average Yield"
        actions['milk_class'] = "moderate"
    else:
        actions['milk'] = "Low transmissible milk quantity. Progeny will exhibit reduced production capacity. Maintain on low-cost forage rations to match margins."
        actions['milk_pill'] = "Low Yield"
        actions['milk_class'] = "attention"
        
    # 3. Reproductive Considerations
    if ptadpr > 1.0:
        actions['repro'] = "Outstanding Daughter Pregnancy Rate transmit potential. Promotes lower calving intervals, reduced vet inputs, and faster postpartum cycles in progeny."
        actions['repro_pill'] = "Elite Fertility"
        actions['repro_class'] = "elite"
    elif ptadpr >= -1.5:
        actions['repro'] = "Baseline reproductive attributes. Calving intervals and breeding services required will match typical herd management standards."
        actions['repro_pill'] = "Standard Fertility"
        actions['repro_class'] = "moderate"
    else:
        actions['repro'] = "Reproductive liability detected. Negative Daughter Pregnancy Rate increases risk of extended cycles. Mate utilizing specialized high-DPR sires."
        actions['repro_pill'] = "Impaired Fertility"
        actions['repro_class'] = "attention"
        
    # 4. Mastitis & Health Risk Indicators
    if ptascs < 2.90:
        actions['health'] = "Elite somatic profile indicates strong natural resistance to mastitis pathogens. Translates to lower clinical incidence and raw milk premiums."
        actions['health_pill'] = "Low Risk"
        actions['health_class'] = "elite"
    elif ptascs <= 3.25:
        actions['health'] = "Average somatic load prediction. Standard milking sanitation and udder preparation routines will control environmental hazards."
        actions['health_pill'] = "Moderate Risk"
        actions['health_class'] = "moderate"
    else:
        actions['health'] = "Increased vulnerability to environmental pathogens. Higher genetic somatic score baseline. Focus on intensive postpartum sanitation and dips."
        actions['health_pill'] = "High Risk"
        actions['health_class'] = "attention"

    # 5. Productive Life
    if ptapl > 2.5:
        actions['longevity'] = "Superb genetic stayability. Anticipated to complete 4+ profitable lactations, maximizing lifetime capital ROI."
        actions['longevity_pill'] = "Elite Longevity"
        actions['longevity_class'] = "elite"
    elif ptapl >= -1.0:
        actions['longevity'] = "Standard herd tenure index. Expected to stay in the milking string for average durations (~3 lactations) before culling."
        actions['longevity_pill'] = "Standard Longevity"
        actions['longevity_class'] = "moderate"
    else:
        actions['longevity'] = "Truncated lifespan index. High risk of early voluntary or involuntary culling due to metabolic deficiencies or lameness genetics."
        actions['longevity_pill'] = "Reduced Lifespan"
        actions['longevity_class'] = "attention"
        
    return actions

# ----------------------------------------------------
# 5. ReportLab PDF Generation
# ----------------------------------------------------
def generate_pdf_report(cow_data, predictions, recommendations):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor('#0F172A')
    accent_blue = colors.HexColor('#2563EB')
    text_dark = colors.HexColor('#020617')
    bg_light = colors.HexColor('#F8FAFC')
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=accent_blue,
        spaceAfter=3
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=text_dark,
        leading=11
    )
    
    white_bold_hdr = ParagraphStyle(
        'WhiteBoldHdr',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )
    
    rec_lbl_style = ParagraphStyle(
        'RecLbl',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=accent_blue
    )
    
    story.append(Paragraph("GENMILK AI - GENOMIC BREEDING & EVALUATION REPORT", title_style))
    story.append(Paragraph("AI-Driven Genomic Prediction Platform for Dairy Production Optimization", body_style))
    story.append(Spacer(1, 10))
    
    meta_data = [
        [Paragraph("<b>Evaluation Timestamp:</b>", body_style), Paragraph(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), body_style),
         Paragraph("<b>Breed Code:</b>", body_style), Paragraph("Holstein-Friesian (HO)", body_style)],
        [Paragraph("<b>Cohort / Year of Birth:</b>", body_style), Paragraph(str(cow_data['yob']), body_style),
         Paragraph("<b>Reliability Index (REL):</b>", body_style), Paragraph(f"{cow_data['REL']:.1f}%", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[120, 150, 100, 170])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. GENOMIC BREEDING VALUES (PTA INPUT MATRIX)", h2_style))
    param_data = [
        [Paragraph("Genomic Trait Indicator", white_bold_hdr), Paragraph("PTA Level", white_bold_hdr), Paragraph("Standard Reference Range", white_bold_hdr), Paragraph("Functional Meaning", white_bold_hdr)],
        [Paragraph("PTA Milk (ptam)", body_style), Paragraph(f"{cow_data['ptam']:.1f} lbs", body_style), Paragraph("-2500 to +1000 lbs", body_style), Paragraph("Estimated transmission capability for milk volume", body_style)],
        [Paragraph("PTA Fat (ptaf)", body_style), Paragraph(f"{cow_data['ptaf']:.1f} lbs", body_style), Paragraph("-100 to +50 lbs", body_style), Paragraph("Estimated transmission capability for milk lipid weight", body_style)],
        [Paragraph("PTA Protein (ptap)", body_style), Paragraph(f"{cow_data['ptap']:.1f} lbs", body_style), Paragraph("-100 to +50 lbs", body_style), Paragraph("Estimated transmission capability for milk protein weight", body_style)],
        [Paragraph("PTA Productive Life (ptapl)", body_style), Paragraph(f"{cow_data['ptapl']:.2f} mo", body_style), Paragraph("-5 to +10 months", body_style), Paragraph("Genetic deviation score for herd culling longevity", body_style)],
        [Paragraph("PTA Daughter Pregnancy Rate (ptadpr)", body_style), Paragraph(f"{cow_data['ptadpr']:.2f}%", body_style), Paragraph("-5% to +5%", body_style), Paragraph("Genetic deviation score for female reproductive rate", body_style)],
        [Paragraph("PTA Somatic Cell Score (ptascs)", body_style), Paragraph(f"{cow_data['ptascs']:.2f}", body_style), Paragraph("2.0 to 4.5 (lower is better)", body_style), Paragraph("Transmissible indicator for immunological udder resistance", body_style)]
    ]
    t_param = Table(param_data, colWidths=[160, 80, 130, 170])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_param)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. PREDICTIVE PHENOTYPIC PROJECTIONS (ML SYSTEM OUTPUTS)", h2_style))
    proj_data = [
        [Paragraph("Realized Production Target", white_bold_hdr), Paragraph("Model Prediction", white_bold_hdr), Paragraph("Confidence & System Framework Reference", white_bold_hdr)],
        [Paragraph("Predicted Lactation Milk Yield", body_style), Paragraph(f"<b>{predictions['milk_yield']:,.0f} lbs</b>", body_style), Paragraph("Random Forest Regressor (R²: 94.68%)", body_style)],
        [Paragraph("Predicted Lactation Fat Yield", body_style), Paragraph(f"<b>{predictions['fat_yield']:.1f} lbs</b>", body_style), Paragraph("Random Forest Regressor (R²: 94.82%)", body_style)],
        [Paragraph("Predicted Lactation Protein Yield", body_style), Paragraph(f"<b>{predictions['protein_yield']:.1f} lbs</b>", body_style), Paragraph("Random Forest Regressor (R²: 96.11%)", body_style)],
        [Paragraph("Genetic Potential Score (GPS)", body_style), Paragraph(f"<b>{predictions['gps']:.1f} / 100</b>", body_style), Paragraph(f"Industry Class: <b>{predictions['category']}</b>", body_style)]
    ]
    t_proj = Table(proj_data, colWidths=[200, 140, 200])
    t_proj.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), accent_blue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_proj)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("3. GENOME-DRIVEN BREEDING STRATEGIES & PROTOCOLS", h2_style))
    recs_list = [
        [Paragraph("Decision Area", white_bold_hdr), Paragraph("Breeding Strategic Guidelines", white_bold_hdr)],
        [Paragraph("Breeding Program Allocation", rec_lbl_style), Paragraph(recommendations['breeding'], body_style)],
        [Paragraph("Lactation Feed Optimization", rec_lbl_style), Paragraph(recommendations['milk'], body_style)],
        [Paragraph("Reproductive Cycle Planning", rec_lbl_style), Paragraph(recommendations['repro'], body_style)],
        [Paragraph("Udder Sanitation Protocols", rec_lbl_style), Paragraph(recommendations['health'], body_style)],
        [Paragraph("Lifespan / Stayability Strategy", rec_lbl_style), Paragraph(recommendations['longevity'], body_style)]
    ]
    t_recs = Table(recs_list, colWidths=[150, 390])
    t_recs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#334155')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,1), (0,-1), bg_light),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_recs)
    story.append(Spacer(1, 16))
    
    story.append(Paragraph("<i>Disclaimer: These predictions are generated from multi-target Random Forest regression estimators trained under historical Holstein dairy databases. Projections represent genetic potentials; actual physical yields are heavily mediated by environment, seasonal temperature, disease outbreaks, forage quality, and farm culling criteria.</i>", ParagraphStyle('Disc', parent=body_style, fontSize=6.2, leading=8.0, textColor=colors.gray)))
    
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.splitlines())

# Helper function to render a progress indicator bar in cards
def draw_percentile_bar(pct, color="#2563EB"):
    return clean_html(f"""
    <div style="width: 100%; background: rgba(255, 255, 255, 0.05); height: 4px; border-radius: 2px; margin-top: 6px; overflow: hidden;">
        <div style="width: {pct:.1f}%; background: {color}; height: 100%; border-radius: 2px;"></div>
    </div>
    """)

# Helper function to render SVG radial progress indicators
def draw_svg_gauge(score, label, max_val=100, unit="", color="#2563EB", size=90):
    percentage = (score / max_val) * 100
    stroke_dashoffset = 251.3 - (251.3 * percentage / 100)
    val_str = f"{score:.1f}" if isinstance(score, float) else f"{score}"
    
    return clean_html(f"""
    <div class="glass-card" style="align-items: center; justify-content: center; text-align: center;">
        <div style="position: relative; width: {size}px; height: {size}px; margin-bottom: 2px; display: flex; align-items: center; justify-content: center;">
            <svg width="{size}" height="{size}" viewBox="0 0 100 100" style="position: absolute;">
                <circle cx="50" cy="50" r="40" stroke="rgba(255, 255, 255, 0.05)" stroke-width="6" fill="transparent" />
                <circle cx="50" cy="50" r="40" stroke="{color}" stroke-width="6" fill="transparent"
                        stroke-dasharray="251.3" stroke-dashoffset="{stroke_dashoffset}"
                        stroke-linecap="round" style="transition: stroke-dashoffset 0.8s ease-in-out; transform: rotate(-90deg); transform-origin: 50% 50%;" />
            </svg>
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <span class="gauge-center-text">{val_str}<span style="font-size: 0.65em; color: #94A3B8; font-weight: 500;">{unit}</span></span>
            </div>
        </div>
        <div class="kpi-title" style="margin-top: 6px; margin-bottom: 0px; text-align: center; font-size: 0.65rem !important;">{label}</div>
    </div>
    """)

# Helper function to render a KPI card without causing markdown indentation bugs
def draw_kpi_card(title, value, unit, pct, color):
    pct_bar = draw_percentile_bar(pct, color)
    return clean_html(f"""
    <div class="glass-card">
        <div>
            <div class="kpi-title">{title}</div>
            <div class="kpi-val">{value}<span class="kpi-unit">{unit}</span></div>
        </div>
        <div>
            <div class="kpi-percentile" style="color: {color};">
                <span class="trend-icon-up">&#9652;</span>
                {pct:.1f}% <span class="kpi-percentile-desc">percentile</span>
            </div>
            {pct_bar}
        </div>
    </div>
    """)

# ----------------------------------------------------
# 6. Streamlit Execution Logic
# ----------------------------------------------------
if not res['loaded']:
    st.info("System Initialization Incomplete: Model assets cannot be resolved.")
else:
    # Scale parameters & base predictions setup
    scaler = res['scaler']
    model_milk = res['model_milk']
    model_fat = res['model_fat']
    model_protein = res['model_protein']
    metrics = res['metrics']
    df_m = res['df_m']
    df_f = res['df_f']
    df_p = res['df_p']
    df_sim = res['df_sim']
    
    # Compute breed averages for 2015 cohort for default inputs
    latest_year = int(df_m['yob1'].max()) # 2015
    row_m_latest = df_m[df_m['yob1'] == latest_year].iloc[0]
    row_f_latest = df_f[df_f['yob1'] == latest_year].iloc[0]
    row_p_latest = df_p[df_p['yob1'] == latest_year].iloc[0]
    
    default_ptam = float(row_m_latest['HO_ptam_RegCow_1']) 
    default_ptaf = float(row_f_latest['HO_ptaf_RegCow_1'])
    default_ptap = float(row_p_latest['HO_ptap_RegCow_1'])
    
    # ----------------------------------------------------
    # ----------------------------------------------------
    # SIDEBAR: Navigation & Input Panels
    # ----------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Console Navigation</div>', unsafe_allow_html=True)
        app_mode = st.radio(
            "Navigation",
            ["Individual Genomic Analysis", "Historical Breed Diagnostics"],
            label_visibility="collapsed"
        )
        
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Genomic Input Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Compute instant outcomes from genomic values</div>', unsafe_allow_html=True)
        
        # Group 1: Production Traits
        st.markdown("<div class='sidebar-section-header'>Production Traits</div>", unsafe_allow_html=True)
        ptam = st.slider("PTA Milk (ptam)", -2500, 1000, value=int(default_ptam), step=10, 
                                 help="Transmissible milk quantity (lbs) above base.")
        ptaf = st.slider("PTA Fat (ptaf)", -100, 75, value=int(default_ptaf), step=1,
                                 help="Transmissible milk fat weight (lbs).")
        ptap = st.slider("PTA Protein (ptap)", -100, 75, value=int(default_ptap), step=1,
                                 help="Transmissible milk protein weight (lbs).")
        
        # Group 2: Health & Fertility Traits
        st.markdown("<div class='sidebar-section-header'>Health & Fertility Traits</div>", unsafe_allow_html=True)
        ptadpr = st.slider("PTA Daughter Pregnancy Rate (ptadpr)", -5.0, 5.0, value=0.0, step=0.1,
                                   help="Deviation in daughter pregnancy cycles (%).")
        ptascs = st.slider("PTA Somatic Cell Score (ptascs)", 2.0, 4.5, value=3.05, step=0.05,
                                   help="Genetic proxy score indicating mastitis resistance.")
        
        # Group 3: Longevity Traits
        st.markdown("<div class='sidebar-section-header'>Longevity Traits</div>", unsafe_allow_html=True)
        ptapl = st.slider("PTA Productive Life (ptapl)", -5.0, 15.0, value=1.8, step=0.1,
                                   help="Deviation in herd productive longevity (months).")
        
        # Group 4: Genomic Metadata & Platform configuration
        st.markdown("<div class='sidebar-section-header'>Assay Configuration</div>", unsafe_allow_html=True)
        rel = st.slider("Assay Reliability (REL)", 50, 99, 85, step=1,
                                help="Assay confidence level based on genotype density.")
        yob = st.number_input("Year of Birth Baseline Correction", 1975, 2026, 2015, step=1,
                                      help="Birth year baseline correction.")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        predict_btn = st.button("Predict", type="primary", use_container_width=True)
        
        # Reset button styled
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        if st.button("Reset Platform Defaults", use_container_width=True):
            st.rerun()

    # ----------------------------------------------------
    # HEADER SECTION
    # ----------------------------------------------------
    st.markdown(clean_html("""
    <div style="margin-top: 5px; margin-bottom: 20px;">
        <h1 style="color: #FFFFFF; font-size: 1.95rem; font-weight: 800; margin-bottom: 2px; letter-spacing: -0.03em;">AI-Driven Genomic Prediction Platform</h1>
        <h3 style="color: #E2E8F0; font-size: 1.0rem; font-weight: 500; margin-bottom: 8px; opacity: 0.9;">Predictive Modelling of Milk Production Parameters in Dairy Cattle using AI and Genomics</h3>
        <p style="color: #94A3B8; font-size: 0.8rem; line-height: 1.45; max-width: 950px; margin-bottom: 12px;">
            Leveraging genomic breeding values and machine learning to estimate milk production performance and genetic potential in dairy cattle.
        </p>
        <div class="status-badge-container">
            <span class="status-badge active"><span class="indicator-dot active"></span>AI Model Active</span>
            <span class="status-badge enabled"><span class="indicator-dot enabled"></span>Genomic Analysis Enabled</span>
        </div>
    </div>
    """), unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # INFERENCE PIPELINE
    # ----------------------------------------------------
    features_keys = ['ptam', 'ptaf', 'ptap', 'ptapl', 'ptadpr', 'ptascs', 'REL', 'yob']
    input_vector = pd.DataFrame([[ptam, ptaf, ptap, ptapl, ptadpr, ptascs, rel, yob]], columns=features_keys)
    scaled_vector = scaler.transform(input_vector)
    
    # Predict realized outputs
    pred_milk = float(model_milk.predict(scaled_vector)[0])
    pred_fat = float(model_fat.predict(scaled_vector)[0])
    pred_protein = float(model_protein.predict(scaled_vector)[0])
    
    # Calculate Genetic Potential Score (GPS)
    s_m = np.clip((ptam + 2500.0) / 3800.0, 0.0, 1.0)
    s_f = np.clip((ptaf + 100.0) / 160.0, 0.0, 1.0)
    s_p = np.clip((ptap + 100.0) / 160.0, 0.0, 1.0)
    s_pl = np.clip((ptapl + 5.0) / 15.0, 0.0, 1.0)
    s_dpr = np.clip((ptadpr + 5.0) / 10.0, 0.0, 1.0)
    s_scs = np.clip((4.5 - ptascs) / 2.5, 0.0, 1.0)
    
    s_prod = 0.4 * s_m + 0.3 * s_f + 0.3 * s_p
    s_health = 0.4 * s_pl + 0.3 * s_dpr + 0.3 * s_scs
    gps = float(np.clip(100.0 * (0.55 * s_prod + 0.45 * s_health), 0.0, 100.0))
    
    # Classify Category
    if pred_milk < 19200:
        category = "Low Producer"
    elif pred_milk >= 24200:
        category = "High Producer"
    else:
        category = "Medium Producer"
        
    # Calculate Confidence Score (based on tree variance consensus)
    t_preds = [float(tree.predict(scaled_vector)[0]) for tree in model_milk.estimators_]
    t_std = np.std(t_preds)
    model_certainty = max(0.0, 100.0 - (t_std / pred_milk * 320.0))
    conf_score = int(np.round(0.6 * rel + 0.4 * model_certainty))
    conf_score = min(max(conf_score, 12), 99)
    
    # Get dynamic database percentiles
    pct_milk = float((df_sim['milk_yield'] < pred_milk).mean() * 100)
    pct_fat = float((df_sim['fat_yield'] < pred_fat).mean() * 100)
    pct_protein = float((df_sim['protein_yield'] < pred_protein).mean() * 100)
    
    # Build recommendations dictionary
    recs = get_breeding_recommendations(ptam, ptaf, ptap, ptapl, ptadpr, ptascs, gps, pred_milk)
    
    # Report generation bytes
    cow_data_pdf = {
        'ptam': ptam, 'ptaf': ptaf, 'ptap': ptap,
        'ptapl': ptapl, 'ptadpr': ptadpr, 'ptascs': ptascs,
        'REL': rel, 'yob': yob
    }
    preds_pdf = {
        'milk_yield': pred_milk, 'fat_yield': pred_fat,
        'protein_yield': pred_protein, 'gps': gps, 'category': category
    }
    pdf_report_bytes = generate_pdf_report(cow_data_pdf, preds_pdf, recs)
    
    # ----------------------------------------------------
    # MAIN VIEW EXECUTION
    # ----------------------------------------------------
    if app_mode == "Individual Genomic Analysis":
        # SECTION 1: Prediction Overview
        st.markdown("<div class='panel-subheader'>Prediction Overview</div>", unsafe_allow_html=True)
        
        # Display 5 dynamic cards in 1 row (3 yield numbers + 2 circular indicators)
        kpi_cols = st.columns(5)
        
        with kpi_cols[0]:
            st.markdown(draw_kpi_card("Predicted Milk Yield", f"{pred_milk:,.0f}", "lbs", pct_milk, "#2563EB"), unsafe_allow_html=True)
            
        with kpi_cols[1]:
            st.markdown(draw_kpi_card("Predicted Fat Yield", f"{pred_fat:,.1f}", "lbs", pct_fat, "#06B6D4"), unsafe_allow_html=True)
            
        with kpi_cols[2]:
            st.markdown(draw_kpi_card("Predicted Protein Yield", f"{pred_protein:,.1f}", "lbs", pct_protein, "#10B981"), unsafe_allow_html=True)
            
        # Select active colors
        gps_color = "#10B981" if gps >= 65 else ("#06B6D4" if gps >= 45 else "#EF4444")
        
        with kpi_cols[3]:
            # Genetic Potential Circular progress
            st.markdown(draw_svg_gauge(gps, "Genetic Potential Score", max_val=100, unit="", color=gps_color), unsafe_allow_html=True)
            
        with kpi_cols[4]:
            # Confidence circular progress
            st.markdown(draw_svg_gauge(conf_score, "Confidence Score", max_val=100, unit="%", color="#2563EB"), unsafe_allow_html=True)
            
        # Middle Grid: Two Columns
        grid_col1, grid_col2 = st.columns([1.05, 0.95])
        
        with grid_col1:
            st.markdown("<div class='panel-subheader'>Genomic Profile Radar Chart</div>", unsafe_allow_html=True)
            
            categories = ['Milk Production', 'Fat Quality', 'Protein Quality', 'Fertility', 'Disease Resistance', 'Longevity']
            values = [s_m * 100, s_f * 100, s_p * 100, s_dpr * 100, s_scs * 100, s_pl * 100]
            
            fig_radar = go.Figure()
            # Feed baseline comparison (100% breed outer baseline, 50% breed median)
            fig_radar.add_trace(go.Scatterpolar(
                r=[50]*6 + [50],
                theta=categories + [categories[0]],
                fill='none',
                line=dict(color='rgba(255, 255, 255, 0.2)', width=1, dash='dash'),
                name='Breed Average Baseline'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(6, 182, 212, 0.12)',
                line=dict(color='#06B6D4', width=2),
                name='Individual Genomics'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        gridcolor='rgba(255,255,255,0.06)',
                        linecolor='rgba(255,255,255,0.06)',
                        tickfont=dict(color='#64748B', size=8),
                        angle=0
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(255,255,255,0.06)',
                        linecolor='rgba(255,255,255,0.06)',
                        tickfont=dict(color='#94A3B8', size=9)
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=10, b=10),
                height=260,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.25,
                    xanchor='center',
                    x=0.5,
                    font=dict(color='#94A3B8', size=8)
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
            
            # SECTION 6: AI Insights Panel & PDF Link
            st.markdown("<div class='panel-subheader'>AI Insights Panel</div>", unsafe_allow_html=True)
            
            # Dynamic AI text calculation inspired by user requests
            milk_eval = "above-average" if ptam > 150 else ("exceptional" if ptam > 450 else "moderate")
            repro_eval = "moderate fertility performance" if ptadpr >= -1.0 else "impaired reproductive attributes"
            if ptadpr > 1.0:
                repro_eval = "high conception indicators"
            long_eval = "strong longevity indicators" if ptapl > 1.5 else "standard stayability factors"
            
            executive_paragraph = f"Animal exhibits {milk_eval} genomic potential for milk production with {repro_eval} and {long_eval}."
            
            st.markdown(clean_html(f"""
            <div class="insight-block">
                {executive_paragraph}
            </div>
            """), unsafe_allow_html=True)
            
            # Export and download report trigger
            st.download_button(
                label="Export Genomic Analysis PDF Report",
                data=pdf_report_bytes,
                file_name=f"GenMilk_Platform_Report_{yob}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with grid_col2:
            st.markdown("<div class='panel-subheader'>Cohort Comparison</div>", unsafe_allow_html=True)
            
            # Breed averages
            avg_milk = float(df_sim['milk_yield'].mean())
            avg_fat = float(df_sim['fat_yield'].mean())
            avg_protein = float(df_sim['protein_yield'].mean())
            
            st.markdown(clean_html(f"""
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Metric Target</th>
                        <th>Predicted Value</th>
                        <th>Breed Average</th>
                        <th>Percentile Ranking</th>
                        <th>Industry Benchmark</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>Milk Volume</b></td>
                        <td class="text-highlight-cyan">{pred_milk:,.0f} lbs</td>
                        <td>{avg_milk:,.0f} lbs</td>
                        <td class="text-highlight-green">{pct_milk:.1f}%</td>
                        <td>24,500 lbs</td>
                    </tr>
                    <tr>
                        <td><b>Fat Components</b></td>
                        <td class="text-highlight-cyan">{pred_fat:.1f} lbs</td>
                        <td>{avg_fat:.1f} lbs</td>
                        <td class="text-highlight-green">{pct_fat:.1f}%</td>
                        <td>950.0 lbs</td>
                    </tr>
                    <tr>
                        <td><b>Protein Components</b></td>
                        <td class="text-highlight-cyan">{pred_protein:.1f} lbs</td>
                        <td>{avg_protein:.1f} lbs</td>
                        <td class="text-highlight-green">{pct_protein:.1f}%</td>
                        <td>760.0 lbs</td>
                    </tr>
                </tbody>
            </table>
            """), unsafe_allow_html=True)
            
            # SECTION 2: Model Explainability Feature Importance
            st.markdown("<div class='panel-subheader'>Model Explainability</div>", unsafe_allow_html=True)
            
            # Mode picker
            target_select = st.radio(
                "Feature Importance Attribution Target:", 
                ["Milk Yield Model", "Fat Yield Model", "Protein Yield Model"], 
                horizontal=True
            )
            
            target_key_map = {
                "Milk Yield Model": "milk_yield",
                "Fat Yield Model": "fat_yield",
                "Protein Yield Model": "protein_yield"
            }
            selected_targ = target_key_map[target_select]
            
            # Chart calculation
            feat_imp_dict = metrics[selected_targ]['feature_importance']
            sorted_imp_items = sorted(feat_imp_dict.items(), key=lambda x: x[1])
            sorted_keys, sorted_vals = zip(*sorted_imp_items)
            
            clean_map = {
                'ptam': 'PTA Milk (ptam)',
                'ptaf': 'PTA Fat (ptaf)',
                'ptap': 'PTA Protein (ptap)',
                'ptapl': 'PTA Productive Life (ptapl)',
                'ptadpr': 'PTA Daughter Pregnancy (ptadpr)',
                'ptascs': 'PTA Somatic Cell (ptascs)',
                'REL': 'Test Reliability (REL)',
                'yob': 'Year of Birth Baseline'
            }
            clean_feature_labels = [clean_map[key] for key in sorted_keys]
            
            bar_col = '#2563EB' if selected_targ == 'milk_yield' else ('#06B6D4' if selected_targ == 'fat_yield' else '#10B981')
            
            fig_feat_imp = go.Figure(go.Bar(
                x=sorted_vals,
                y=clean_feature_labels,
                orientation='h',
                marker=dict(
                    color=bar_col,
                    line=dict(width=0)
                ),
                width=0.5
            ))
            
            fig_feat_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.05)',
                    tickfont=dict(color='#94A3B8', size=8),
                    zeroline=False,
                    title='Feature Attention Weight'
                ),
                yaxis=dict(
                    tickfont=dict(color='#F8FAFC', size=8)
                ),
                margin=dict(l=10, r=10, t=5, b=5),
                height=160,
                showlegend=False
            )
            st.plotly_chart(fig_feat_imp, use_container_width=True, config={'displayModeBar': False})
            
        # SECTION 5: Breeding Recommendation Engine (2x2 Layout)
        st.markdown("<div class='panel-subheader'>Breeding Recommendation Engine & Strategy</div>", unsafe_allow_html=True)
        
        rec_cols = st.columns(2)
        with rec_cols[0]:
            st.markdown(clean_html(f"""
            <div class="rec-box border-{recs['breeding_class']}">
                <div class="rec-header">
                    <span>Selective Breeding Suitability</span>
                    <span class="rec-status-pill pill-{recs['breeding_class']}">{recs['breeding_pill']}</span>
                </div>
                <div class="rec-body">{recs['breeding']}</div>
            </div>
            <div class="rec-box border-{recs['milk_class']}">
                <div class="rec-header">
                    <span>Milk Optimization Strategy</span>
                    <span class="rec-status-pill pill-{recs['milk_class']}">{recs['milk_pill']}</span>
                </div>
                <div class="rec-body">{recs['milk']}</div>
            </div>
            """), unsafe_allow_html=True)
            
        with rec_cols[1]:
            st.markdown(clean_html(f"""
            <div class="rec-box border-{recs['repro_class']}">
                <div class="rec-header">
                    <span>Fertility Considerations</span>
                    <span class="rec-status-pill pill-{recs['repro_class']}">{recs['repro_pill']}</span>
                </div>
                <div class="rec-body">{recs['repro']}</div>
            </div>
            <div class="rec-box border-{recs['health_class']}">
                <div class="rec-header">
                    <span>Health Risk Indicators</span>
                    <span class="rec-status-pill pill-{recs['health_class']}">{recs['health_pill']}</span>
                </div>
                <div class="rec-body">{recs['health']}</div>
            </div>
            """), unsafe_allow_html=True)

        # Optional, self-contained clinical decision-support module.  It uses no
        # production-model outputs or pickle assets, so the established genomic
        # prediction workflow remains unchanged unless this section is opened.
        with st.expander("Disease Risk Prediction", expanded=False):
            render_disease_risk_module(ptascs, ptapl, ptadpr)
            
    else:
        st.markdown("<div class='panel-subheader'>Historical Breed Diagnostics</div>", unsafe_allow_html=True)
        st.write("Plotting breed genetic trajectories and components over 40 years of Holstein records to compare modern selections against historical baselines.")
        
        # Plotly Historical Chart
        fig_historic = go.Figure()
        
        # PTA Milk
        fig_historic.add_trace(go.Scatter(
            x=df_m['yob1'],
            y=df_m['HO_ptam_RegCow_1'],
            mode='lines+markers',
            name='PTA Milk (Left Axis)',
            line=dict(color='#2563EB', width=2.5),
            marker=dict(
                size=8,
                color='#2563EB',
                line=dict(color='#0F172A', width=2),
                opacity=0.9
            )
        ))
        
        # PTA Fat
        fig_historic.add_trace(go.Scatter(
            x=df_f['yob1'],
            y=df_f['HO_ptaf_RegCow_1'],
            mode='lines',
            name='PTA Fat (Right Axis)',
            line=dict(color='#06B6D4', width=1.5, dash='dash'),
            yaxis='y2'
        ))
        
        # PTA Protein
        fig_historic.add_trace(go.Scatter(
            x=df_p['yob1'],
            y=df_p['HO_ptap_RegCow_1'],
            mode='lines',
            name='PTA Protein (Right Axis)',
            line=dict(color='#10B981', width=1.5, dash='dot'),
            yaxis='y2'
        ))
        
        fig_historic.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.05)',
                tickfont=dict(color='#94A3B8', size=9),
                title='Cohort Birth Year'
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.05)',
                tickfont=dict(color='#2563EB', size=8),
                title='PTA Milk (lbs)'
            ),
            yaxis2=dict(
                tickfont=dict(color='#06B6D4', size=8),
                title='PTA Components (lbs)',
                overlaying='y',
                side='right'
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(color='#94A3B8', size=8)
            )
        )
        
        st.plotly_chart(fig_historic, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        **Chronological Diagnostics:**
        - **Historical Progress:** Breeding values reflect differences relative to a baseline reference period. The rapid ascending trajectory of PTA Milk (from `-2228.3 lbs` in 1975 to `+406.3 lbs` in 2015) illustrates standard cumulative genetic progress under active selection pressure.
        - **Components Shift:** Lipid and protein transmitting abilities have grown in direct proportion, reflecting multi-target selection indexes (e.g. Net Merit) deployed across modern agritech and commercial herds.
        """)
