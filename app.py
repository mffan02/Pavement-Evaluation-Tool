import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Set page config
st.set_page_config(page_title="Pavement Condition Evaluation Tool", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .header {font-size: 2.5rem; font-weight: bold; color: #0066cc;}
</style>
""", unsafe_allow_html=True)

# ============ FUNCTIONS ============

# PCI Parameters (From Excel/JKR Standards)
DEFECT_WEIGHTS = {
    "Longitudinal Crack": 1.0,
    "Alligator (Fatigue) Crack": 1.6,
    "Potholes": 2.2,
    "Raveling": 1.2,
    "Depression/Sag": 1.4,
    "Patching (Failed)": 1.8,
    "Bleeding/Flushing": 1.0,
    "Rut/Rutting": 1.6
}

SEVERITY_FACTORS = {
    "Low": 0.6,
    "Medium": 1.0,
    "High": 1.4
}

def calculate_pci_single(area, defect_type, severity):
    """Calculate PCI for a single defect entry"""
    weighting_factor = DEFECT_WEIGHTS.get(defect_type, 0)
    severity_factor = SEVERITY_FACTORS.get(severity, 0)
    # Deduct Value (DV) – Excel formula
    dv = (area / 100) * weighting_factor * severity_factor * 100
    pci = max(0, 100 - dv)
    return round(pci, 2)

def pci_condition(pci):
    """Determine condition and maintenance based on PCI value"""
    if pci >= 85:
        return "Very Good", "Routine maintenance"
    elif pci >= 70:
        return "Good / Satisfactory", "Preventive maintenance"
    elif pci >= 55:
        return "Fair", "Surface treatment / Overlay"
    else:
        return "Poor", "Major rehabilitation / Reconstruction"

def calculate_pci(defects_df):
    """Calculate Pavement Condition Index (PCI) from defects - Standard Method"""
    if defects_df.empty:
        return 100
    
    # Updated severity weights - accepts both short and long forms
    severity_weights = {
        'L': 1, 'Low': 1,
        'M': 3, 'Medium': 3,
        'H': 5, 'High': 5
    }
    
    defect_factors = {
        'Alligator Cracking': 8,
        'Linear Cracking': 4,
        'Potholes': 9,
        'Rutting': 5,
        'Raveling': 3,
        'Bleeding': 2,
        'Other': 3
    }
    
    total_deduct_value = 0
    
    for idx, row in defects_df.iterrows():
        defect_type = row.get('Defect Type', 'Other')
        severity = row.get('Severity', 'Low')
        area_pct = row.get('Area Percentage (%)', 
                   row.get('Area Affected (%)', 0))
        
        base_factor = defect_factors.get(defect_type, 3)
        severity_weight = severity_weights.get(severity, 1)
        
        deduct = (base_factor * severity_weight * area_pct) / 100
        total_deduct_value += deduct
    
    pci = max(0, min(100, 100 - total_deduct_value))
    return pci

def calculate_pci_simplified(section_df):
    """
    Simplified PCI model adapted for academic / JKR-style assessment
    Uses direct penalty approach based on severity and affected area
    """
    if section_df.empty:
        return 100
    
    base_pci = 100
    severity_penalty = {
        "L": 2,
        "Low": 2,
        "M": 5,
        "Medium": 5,
        "H": 10,
        "High": 10
    }
    
    total_penalty = 0
    for _, row in section_df.iterrows():
        severity = row.get("Severity", "Low")
        # Try both column names for area
        area = row.get("Area Affected (%)", 
               row.get("Area Percentage (%)", 0))
        
        penalty = severity_penalty.get(severity, 2) * (area / 10)
        total_penalty += penalty
    
    pci = base_pci - total_penalty
    return max(0, round(pci, 0))

def classify_condition(pci):
    """Classify pavement condition based on PCI"""
    if pci >= 85:
        return 'Very Good', '#2ecc71'
    elif pci >= 70:
        return 'Good', '#3498db'
    elif pci >= 55:
        return 'Fair', '#f39c12'
    else:
        return 'Poor', '#e74c3c'

def classify_pci(pci):
    """Classify PCI value into categories"""
    if pci >= 85:
        return "Very Good"
    elif pci >= 70:
        return "Good"
    elif pci >= 55:
        return "Fair"
    else:
        return "Poor"

def classify_iri(iri):
    """Classify IRI value into categories"""
    if iri is None or pd.isna(iri):
        return "N/A"
    if iri <= 2.0:
        return "Very Good"
    elif iri <= 4.0:
        return "Good"
    elif iri <= 6.0:
        return "Fair"
    else:
        return "Poor"

def maintenance_decision(pci_class, iri_class):
    """Determine maintenance action based on PCI and IRI classifications"""
    if iri_class == "N/A":
        # Fallback to PCI-only decision
        if pci_class == "Very Good":
            return "Routine/Preventive Maintenance (Crack Sealing)"
        elif pci_class == "Good":
            return "Routine Maintenance (Pothole Repair, Crack Sealing)"
        elif pci_class == "Fair":
            return "Corrective Maintenance (Mill and Overlay, Patching)"
        else:
            return "Major Rehabilitation (Full Depth Reclaim, Thick Overlay)"
    
    # Combined PCI and IRI decision
    if pci_class == "Very Good" and iri_class == "Very Good":
        return "Routine/Preventive Maintenance (Crack Sealing)"
    if pci_class == "Very Good" and iri_class in ["Good", "Fair"]:
        return "Minor Rehabilitation (Surface Treatment/Thin Overlay)"
    if pci_class == "Good" and iri_class in ["Fair", "Poor"]:
        return "Major Rehabilitation (Medium Overlay/Recycling)"
    if pci_class in ["Fair", "Poor"] or iri_class == "Poor":
        return "Reconstruction/Heavy Overlay (Structural Repair)"
    return "Preventive Maintenance"

# ============ MAIN APP ============

st.markdown('<p class="header">🛣️ Digital Pavement Condition Evaluation Tool</p>', unsafe_allow_html=True)
st.markdown("**District JKR Maintenance Division - Pavement Asset Management System**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.info("""
    **How to use:**
    
    1. Upload Excel file with columns:
       - Section ID
       - Defect Type
       - Severity (Low/Medium/High or L/M/H)
       - Area Percentage (%) or Area Affected (%)
       - IRI (optional - case insensitive)
    
    2. View automatic calculations
    3. Download results
    """)
    
    st.markdown("---")
    st.subheader("⚙️ Calculation Method")
    pci_method = st.radio(
        "Select PCI Calculation:",
        ["Standard Method", "Simplified Method (JKR)"],
        help="Standard: Uses defect type factors\nSimplified: Direct penalty based on severity"
    )
    
    st.markdown("---")
    st.subheader("📥 Sample Data Format")
    sample_data = {
        'Section ID': ['SEC-001', 'SEC-002', 'SEC-003'],
        'Defect Type': ['Alligator Cracking', 'Potholes', 'Linear Cracking'],
        'Severity': ['Medium', 'High', 'Low'],
        'Area Percentage (%)': [15, 5, 8],
        'IRI': [3.2, 4.5, 2.8]
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload Data", "🧮 PCI Calculator", "📊 Analysis", "📈 Dashboard", "📄 Report"])

# TAB 1: Upload
with tab1:
    st.subheader("Upload Pavement Condition Data")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            st.success("✅ File uploaded successfully!")
            st.dataframe(df, use_container_width=True)
            st.session_state.data = df
        except Exception as e:
            st.error(f"Error reading file: {e}")

# TAB 2: PCI Calculator
# TAB 3: Analysis (previously TAB 2)
with tab3:
    st.subheader("🧮 Pavement Condition Index (PCI) Calculator")
    st.markdown("**Single Defect PCI Evaluation Tool**")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Input Parameters")
        
        area = st.number_input(
            "Defect Area (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=10.0,
            step=1.0,
            help="Enter the percentage of pavement area affected by the defect"
        )
        
        defect_type = st.selectbox(
            "Defect Type",
            list(DEFECT_WEIGHTS.keys()),
            help="Select the type of pavement defect"
        )
        
        severity = st.selectbox(
            "Severity Level",
            list(SEVERITY_FACTORS.keys()),
            help="Low: Minor defects | Medium: Moderate defects | High: Severe defects"
        )
        
        st.markdown("---")
        
        if st.button("🔍 Calculate PCI", type="primary", use_container_width=True):
            pci_value = calculate_pci_single(area, defect_type, severity)
            condition, maintenance = pci_condition(pci_value)
            
            # Store in session state for charts
            st.session_state.pci_calc = {
                'pci': pci_value,
                'condition': condition,
                'maintenance': maintenance,
                'area': area,
                'defect': defect_type,
                'severity': severity
            }
    
    with col2:
        st.markdown("#### Weight Factors")
        st.markdown("**Defect Weights:**")
        for defect, weight in DEFECT_WEIGHTS.items():
            st.text(f"• {defect}: {weight}")
        
        st.markdown("**Severity Factors:**")
        for sev, factor in SEVERITY_FACTORS.items():
            st.text(f"• {sev}: {factor}")
    
    # Display results if calculation has been done
    if 'pci_calc' in st.session_state:
        calc = st.session_state.pci_calc
        
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("PCI Value", f"{calc['pci']}", delta=None)
        
        with col2:
            # Color code the condition
            condition_color = {
                "Very Good": "🟢",
                "Good / Satisfactory": "🔵", 
                "Fair": "🟠",
                "Poor": "🔴"
            }
            icon = condition_color.get(calc['condition'], "⚪")
            st.metric("Condition", f"{icon} {calc['condition']}")
        
        with col3:
            st.metric("Area Affected", f"{calc['area']}%")
        
        # Maintenance recommendation
        st.info(f"**Recommended Maintenance:** {calc['maintenance']}")
        
        # Visual charts
        st.markdown("---")
        st.markdown("### 📈 PCI Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PCI Value Chart
            pci_data = pd.DataFrame({
                "Metric": ["Current PCI"],
                "Value": [calc['pci']]
            })
            
            fig_pci_single = px.bar(
                pci_data,
                x="Metric",
                y="Value",
                title=f"Calculated PCI: {calc['pci']}",
                color="Value",
                color_continuous_scale=["red", "orange", "yellow", "green"],
                range_color=[0, 100]
            )
            fig_pci_single.update_layout(showlegend=False, yaxis_range=[0, 100])
            fig_pci_single.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Very Good")
            fig_pci_single.add_hline(y=70, line_dash="dash", line_color="blue", annotation_text="Good")
            fig_pci_single.add_hline(y=55, line_dash="dash", line_color="orange", annotation_text="Fair")
            st.plotly_chart(fig_pci_single, use_container_width=True)
        
        with col2:
            # PCI Condition Bands
            pci_bands = pd.DataFrame({
                "Condition": ["Very Good (85-100)", "Good (70-84)", "Fair (55-69)", "Poor (0-54)"],
                "Lower Bound": [85, 70, 55, 0],
                "Upper Bound": [100, 84, 69, 54],
                "Color": ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
            })
            
            fig_bands = go.Figure()
            
            # Add current PCI marker
            fig_bands.add_trace(go.Scatter(
                x=[calc['pci']],
                y=[1],
                mode='markers',
                marker=dict(size=20, color='red', symbol='diamond'),
                name='Current PCI',
                showlegend=True
            ))
            
            # Add condition bands
            for idx, row in pci_bands.iterrows():
                fig_bands.add_shape(
                    type="rect",
                    x0=row["Lower Bound"], x1=row["Upper Bound"],
                    y0=0, y1=2,
                    fillcolor=row["Color"],
                    opacity=0.3,
                    line_width=0
                )
                fig_bands.add_annotation(
                    x=(row["Lower Bound"] + row["Upper Bound"]) / 2,
                    y=1.5,
                    text=row["Condition"].split()[0],
                    showarrow=False,
                    font=dict(size=10)
                )
            
            fig_bands.update_layout(
                title="PCI Position on Condition Scale",
                xaxis_title="PCI Value",
                xaxis_range=[0, 100],
                yaxis_visible=False,
                height=400
            )
            
            st.plotly_chart(fig_bands, use_container_width=True)
        
        # Summary table
        st.markdown("---")
        st.markdown("### 📋 Calculation Summary")
        
        summary_df = pd.DataFrame({
            "Parameter": ["Defect Type", "Severity Level", "Area Affected", "Defect Weight", "Severity Factor", "Deduct Value", "PCI Score", "Condition Class", "Maintenance Action"],
            "Value": [
                calc['defect'],
                calc['severity'],
                f"{calc['area']}%",
                DEFECT_WEIGHTS.get(calc['defect'], 0),
                SEVERITY_FACTORS.get(calc['severity'], 0),
                f"{100 - calc['pci']:.2f}",
                calc['pci'],
                calc['condition'],
                calc['maintenance']
            ]
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# TAB 3: Analysis (previously TAB 2)
with tab2:
    st.subheader("Condition Analysis Results")
    
    if 'data' in st.session_state:
        df = st.session_state.data
        
        results = []
        for section in df['Section ID'].unique():
            section_data = df[df['Section ID'] == section]
            
            # Calculate PCI
            pci = calculate_pci(section_data)
            pci_class = classify_pci(pci)
            condition, color = classify_condition(pci)
            
            # Flexible IRI column reading - case insensitive
            iri_col = None
            for col in section_data.columns:
                if col.strip().lower().startswith('iri'):
                    iri_col = col
                    break
            
            # Calculate IRI
            iri = section_data[iri_col].mean() if iri_col else None
            iri_class = classify_iri(iri)
            
            # Determine maintenance action based on both PCI and IRI
            maintenance = maintenance_decision(pci_class, iri_class)
            
            results.append({
                'Section ID': section,
                'PCI': round(pci, 2),
                'Condition': pci_class,
                'IRI': round(iri, 2) if iri else 'N/A',
                'IRI Classification': iri_class,
                'Maintenance Action': maintenance
            })
        
        results_df = pd.DataFrame(results)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_pci = results_df['PCI'].mean()
            st.metric("Average PCI", f"{avg_pci:.1f}")
        with col2:
            st.metric("Sections Analyzed", len(results_df))
        with col3:
            good_sections = len(results_df[results_df['Condition'].isin(['Very Good', 'Good'])])
            st.metric("Good/Excellent", f"{good_sections}/{len(results_df)}")
        with col4:
            poor_sections = len(results_df[results_df['Condition'].isin(['Fair', 'Poor'])])
            st.metric("Fair/Poor", f"{poor_sections}/{len(results_df)}")
        
        st.markdown("---")
        st.dataframe(results_df, use_container_width=True)
        st.session_state.results = results_df
    else:
        st.warning("⚠️ Please upload data first")

# TAB 4: Dashboard (previously TAB 3)
with tab4:
    st.subheader("IRI Performance Dashboard")
    
    if 'results' in st.session_state:
        results_df = st.session_state.results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter out N/A IRI values for the chart
            iri_df = results_df[results_df['IRI'] != 'N/A'].copy()
            if not iri_df.empty:
                iri_df['IRI'] = pd.to_numeric(iri_df['IRI'], errors='coerce')
                fig_iri = px.line(
                    iri_df,
                    x="Section ID",
                    y="IRI",
                    markers=True,
                    title="International Roughness Index (IRI) per Road Section"
                )
                fig_iri.add_hline(y=4.0, line_dash="dash", line_color="red", annotation_text="JKR Threshold (4.0 m/km)")
                fig_iri.update_layout(
                    xaxis_title="Section ID",
                    yaxis_title="IRI (m/km)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_iri, use_container_width=True)
            else:
                st.warning("⚠️ No IRI data available to display")
        
        with col2:
            iri_counts = results_df["IRI Classification"].value_counts()
            fig_pie = px.pie(
                values=iri_counts.values,
                names=iri_counts.index,
                title="IRI Condition Distribution",
                color_discrete_map={
                    'Very Good': '#2ecc71',
                    'Good': '#3498db',
                    'Fair': '#f39c12',
                    'Poor': '#e74c3c',
                    'N/A': '#95a5a6'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("⚠️ Please upload and analyse data first")

# TAB 5: Report (previously TAB 4)
with tab5:
    st.subheader("Technical Report")
    
    if 'results' in st.session_state:
        results_df = st.session_state.results
        report_date = datetime.now().strftime("%d/%m/%Y")
        
        st.markdown(f"""
        ## Pavement Condition Evaluation Report
        **Date:** {report_date}  
        **District:** JKR Maintenance Division
        
        ### Summary
        - **Sections:** {len(results_df)}
        - **Avg PCI:** {results_df['PCI'].mean():.1f}
        
        ### Results
        """)
        
        for idx, row in results_df.iterrows():
            st.markdown(f"""
            **{row['Section ID']}**
            - PCI: {row['PCI']} ({row['Condition']})
            - IRI: {row['IRI']} ({row['IRI Classification']})
            - Action: {row['Maintenance Action']}
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, 
                             file_name=f"Results_{datetime.now().strftime('%Y%m%d')}.csv")
        
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False)
            excel_buffer.seek(0)
            st.download_button("📊 Download Excel", excel_buffer.getvalue(),
                             file_name=f"Results_{datetime.now().strftime('%Y%m%d')}.xlsx",
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
