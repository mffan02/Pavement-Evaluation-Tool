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

def calculate_pci(defects_df):
    """Calculate Pavement Condition Index (PCI) from defects"""
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
       - Area Percentage (%)
       - IRI (optional - case insensitive)
    
    2. View automatic calculations
    3. Download results
    """)
    
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
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "📊 Analysis", "📈 Dashboard", "📄 Report"])

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

# TAB 2: Analysis
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

# TAB 3: Dashboard
with tab3:
    st.subheader("Visual Dashboard")
    
    if 'results' in st.session_state:
        results_df = st.session_state.results
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pci = px.bar(results_df, x='Section ID', y='PCI', 
                            color='Condition', title='PCI by Section',
                            color_discrete_map={
                                'Very Good': '#2ecc71', 'Good': '#3498db',
                                'Fair': '#f39c12', 'Poor': '#e74c3c', 'Very Poor': '#c0392b'
                            })
            fig_pci.add_hline(y=70, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pci, use_container_width=True)
        
        with col2:
            condition_counts = results_df['Condition'].value_counts()
            fig_pie = px.pie(values=condition_counts.values, names=condition_counts.index,
                           title='Condition Distribution',
                           color_discrete_map={
                               'Very Good': '#2ecc71', 'Good': '#3498db',
                               'Fair': '#f39c12', 'Poor': '#e74c3c', 'Very Poor': '#c0392b'
                           })
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("⚠️ Run analysis first")

# TAB 4: Report
with tab4:
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
