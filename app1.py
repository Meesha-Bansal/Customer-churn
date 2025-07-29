import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import urllib.parse

# ========== Streamlit Page Config ==========
st.set_page_config(
    page_title="ChurnAI - Customer Prediction Platform",
    page_icon="🎯",
    layout="wide"
)

# ========== CSS ==========
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        body, .main, .block-container { background: linear-gradient(135deg, #1a1d23 0%, #2d3748 50%, #4a5568 100%);}
        .header-title {
            font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #a0aec0 0%, #e2e8f0 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; display: center;
        }
        .upload-zone {
            border: 3px dashed rgba(160, 174, 192, 0.4);
            border-radius: 20px; padding: 3rem; text-align: center;
            background: rgba(74, 85, 104, 0.2); margin: 2rem 0; color: #cbd5e0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 18px; padding: 1.8rem;
            text-align: center; margin: 0.5rem;
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        }
        .metric-value { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
        .metric-label { font-size: 1rem; opacity: 0.9; font-weight: 400; }
        .dashboard-section {
            background: rgba(45, 55, 72, 0.8); border-radius: 20px; padding: 2.5rem;
            margin: 2rem 0; color: #e2e8f0;
        }
    </style>
""", unsafe_allow_html=True)

api_url = "http://127.0.0.1:8000"

def get_recommendation(risk_level):
    recommendations = {
        '🔴 High Risk': 'Immediate intervention required - Deploy retention specialist',
        '🟡 Medium Risk': 'Proactive engagement - Offer loyalty incentives',
        '🟢 Low Risk': 'Maintain satisfaction - Regular check-ins'
    }
    return recommendations.get(risk_level, 'Monitor regularly')

if 'active_section' not in st.session_state:
    st.session_state.active_section = 'single'
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'shap_data' not in st.session_state:
    st.session_state.shap_data = None

# ============ ROUTE SWITCH FOR DASHBOARD ============
query_params = st.query_params
if query_params.get("dashboard", "false") == "true" and st.session_state.get('dashboard_data'):
    dashboard_data = st.session_state.dashboard_data
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
            color: white; text-align: center;
        ">
            <h1 style="margin: 0; color: #f7fafc;">📊 Customer Churn Analytics Dashboard</h1>
            <p style="margin: 0.5rem 0 0 0; color: #cbd5e0; opacity: 0.9;">Executive Summary & Risk Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📈 Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"""<div class="metric-card"><div class="metric-value">{dashboard_data['total_customers']:,}</div><div class="metric-label">👥 Total Customers</div></div>""", unsafe_allow_html=True)
    with col2: st.markdown(f"""<div class="metric-card"><div class="metric-value">{dashboard_data['churned']:,}</div><div class="metric-label">🚨 At-Risk Customers</div></div>""", unsafe_allow_html=True)
    with col3: st.markdown(f"""<div class="metric-card"><div class="metric-value">{dashboard_data['churn_rate']:.1f}%</div><div class="metric-label">📈 Churn Rate</div></div>""", unsafe_allow_html=True)
    with col4: st.markdown(f"""<div class="metric-card"><div class="metric-value">{dashboard_data['avg_probability']:.1f}%</div><div class="metric-label">📊 Avg Risk Score</div></div>""", unsafe_allow_html=True)
    st.markdown("### 📊 Visual Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Customer Risk Distribution")
        risk_data = dashboard_data['risk_distribution']
        fig_pie = px.pie(
            values=list(risk_data.values()),
            names=list(risk_data.keys()),
            title="Risk Level Distribution",
            color_discrete_sequence=['#e53e3e', '#ed8936', '#48bb78']
        )
        fig_pie.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#2d3748'), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown("#### 📊 Churn Probability Distribution")
        fig_hist = px.histogram(
            x=dashboard_data['probability_data'],
            title="Probability Distribution",
            nbins=25,
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#2d3748'), xaxis_title="Churn Probability", yaxis_title="Number of Customers")
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("### 📋 Risk Analysis Summary")
    risk_summary = []
    for risk_level, count in dashboard_data['risk_distribution'].items():
        percentage = (count / dashboard_data['total_customers']) * 100
        risk_summary.append({
            'Risk Level': risk_level,
            'Customer Count': f"{count:,}",
            'Percentage': f"{percentage:.1f}%",
            'Recommended Action': get_recommendation(risk_level)
        })
    summary_df = pd.DataFrame(risk_summary)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("### 💡 Business Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🔴 Immediate Actions Required
        - **High-risk customers** need immediate retention interventions
        - **Personalized offers** should be deployed within 48 hours
        - **Customer success teams** should prioritize outreach
        - **Contract negotiations** for flexible terms
        """)
    with col2:
        st.markdown("""
        #### 🟢 Proactive Strategies
        - **Loyalty programs** for medium-risk customers
        - **Service enhancement** based on usage patterns
        - **Regular check-ins** for relationship building
        - **Value demonstration** through success metrics
        """)
    st.markdown("---")
    if st.button("⬅️ Back to Analysis"):
        st.query_params.clear()  # Clears all query params, returns to Batch Upload
        st.rerun()
    st.stop()

# ========== HEADER AND NAVIGATION ==========
st.markdown('<div class="header-title">🎯 ChurnAI Platform</div>', unsafe_allow_html=True)
st.markdown('<div style="display: flex; gap: 1rem; margin: 2rem 0;">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("👤 Single Prediction", key="nav_single"):
        st.session_state.active_section = 'single'
        st.session_state.analysis_complete = False
        st.session_state.batch_results = None
with col2:
    if st.button("📊 Batch Upload", key="nav_batch"):
        st.session_state.active_section = 'batch'
        st.session_state.analysis_complete = False
        st.session_state.batch_results = None
st.markdown('</div>', unsafe_allow_html=True)

# ======================= SINGLE PREDICTION ======================
if st.session_state.active_section == 'single':
    st.markdown("### 👤 Individual Customer Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👥 Customer Demographics")
        gender_val = 0 if st.selectbox("Gender", ["Male", "Female"], key="gender_single") == "Male" else 1
        senior_val = 1 if st.selectbox("Senior Citizen Status", ["No", "Yes"], key="senior_single") == "Yes" else 0
        partner_val = 1 if st.selectbox("Has Partner", ["No", "Yes"], key="partner_single") == "Yes" else 0
        tenure = st.number_input("Tenure (months)", min_value=0.0, max_value=100.0, value=12.0, key="tenure_single")
    with col2:
        st.markdown("#### 💰 Billing Information")
        monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, key="monthly_single")
        totalcharges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0, key="total_single")
        contract_val = {"Month-to-month": 0, "One year": 1, "Two year+": 2}[
            st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year+"], key="contract_single")
        ]
    st.markdown("#### 🌐 Service Portfolio")
    col3, col4 = st.columns(2)
    with col3:
        services = st.multiselect("Communication Services",
            ["Phone Service", "Internet Service"], key="services_single")
        phoneservice_val = len(services)
    with col4:
        online_service = st.selectbox("Premium Online Services",
            ["None", "Online Security", "Online Backup", "Tech Support", "Device Protection"], key="online_single")
        onlineservice_val = 0 if online_service == "None" else 1
    streaming_service = st.selectbox("Entertainment Streaming",
        ["None", "Streaming Movies", "Streaming TV", "Streaming Music"], key="streaming_single")
    streaming_val = 0 if streaming_service == "None" else 1

    if st.button("🚀 Analyze Customer Churn Risk", key="predict_single_btn"):
        with st.spinner("🔮 Processing customer data with AI..."):
            payload = {
                "gender": gender_val,
                "seniorcitizen": senior_val,
                "partner": partner_val,
                "tenure": tenure,
                "phoneservice": phoneservice_val,
                "onlineservice": onlineservice_val,
                "streaming": streaming_val,
                "contract": contract_val,
                "monthlycharges": monthlycharges,
                "totalcharges": totalcharges
            }
            try:
                response = requests.post(f"{api_url}/predict", json=payload)
                result = response.json()
                if response.status_code == 200:
                    churn_class = "churn" if result['prediction'] == 1 else ""
                    prediction_text = "⚠️ HIGH CHURN RISK" if result['prediction'] == 1 else "✅ CUSTOMER RETAINED"
                    probability = result['probability'] * 100
                    st.markdown(f"""
                        <div class="prediction-result {churn_class}">
                            <h1>{prediction_text}</h1>
                            <div class="metric-value">{probability:.1f}%</div>
                            <div class="metric-label">Churn Probability Score</div>
                        </div>
                    """, unsafe_allow_html=True)
                    if probability > 70:
                        risk_level = "🔴 CRITICAL RISK"
                        risk_color = "linear-gradient(135deg, #e53e3e 0%, #c53030 100%)"
                    elif probability > 40:
                        risk_level = "🟡 MODERATE RISK"
                        risk_color = "linear-gradient(135deg, #ed8936 0%, #dd6b20 100%)"
                    else:
                        risk_level = "🟢 LOW RISK"
                        risk_color = "linear-gradient(135deg, #48bb78 0%, #38a169 100%)"
                    st.markdown(f"""
                        <div style="background: {risk_color}; color: white; padding: 1.5rem; 
                                   border-radius: 15px; text-align: center; margin: 1.5rem 0;
                                   box-shadow: 0 15px 35px rgba(0,0,0,0.3); font-size: 1.3rem; font-weight: 600;">
                            Risk Assessment: {risk_level}
                        </div>
                    """, unsafe_allow_html=True)
                    if 'shap' in result:
                        st.markdown('<div class="shap-section">', unsafe_allow_html=True)
                        st.markdown("### 🔬 AI Model Explainability (SHAP Analysis)")
                        shap_df = pd.DataFrame(result['shap'].items(), columns=["Feature", "Impact"])
                        shap_df = shap_df.sort_values(by="Impact", key=lambda x: x.abs(), ascending=False)
                        st.markdown("#### 🎯 Key Decision Factors")
                        for i, row in shap_df.head(5).iterrows():
                            direction = "increases" if row['Impact'] > 0 else "reduces"
                            icon = "📈" if row['Impact'] > 0 else "📉"
                            st.markdown(f"""
                                <div class="feature-impact">
                                    {icon} <strong>{row['Feature']}</strong> {direction} churn probability by <strong>{abs(row['Impact']):.3f}</strong>
                                </div>
                            """, unsafe_allow_html=True)
                        fig = px.bar(
                            shap_df.head(8),
                            x='Impact',
                            y='Feature',
                            orientation='h',
                            color='Impact',
                            color_continuous_scale=['#ed8936', '#2d3748', '#48bb78'],
                            title="🔍 Feature Impact Analysis"
                        )
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e2e8f0'),
                            title_font_color='#f7fafc'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.success("✨ Analysis complete! SHAP explainability displayed below.")
                else:
                    st.error(f"❌ Prediction Error: {result.get('error', 'Unknown error occurred')}")
            except Exception as e:
                st.error(f"🔌 Connection Error: {str(e)}")

# =================== BATCH UPLOAD ========================
if st.session_state.active_section == 'batch':
    st.markdown("### 📊 Batch Customer Intelligence")
    # st.markdown("""
    #     <div class="upload-zone">
    #         <h2>📁 Upload Customer Dataset</h2>
    #         <p>Drop your CSV file here for comprehensive batch analysis</p>
    #         <small>Supported format: CSV with customer attributes</small>
    #     </div>
    # """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📎 Select or drag your CSV file here",
        type=['csv'],
        key="batch_upload",
        help="Supports drag & drop functionality - just drop your CSV file above!",
        label_visibility="collapsed"
    )

    if uploaded_file and not st.session_state.analysis_complete:
        st.success(f"✅ Dataset loaded: {uploaded_file.name}")
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.markdown("#### 👀 Dataset Preview")
            st.dataframe(preview_df.head(8), use_container_width=True)
            if st.button("🚀 Execute Batch Analysis", key="batch_analyze"):
                with st.spinner("⚡ Processing batch predictions with AI..."):
                    try:
                        uploaded_file.seek(0)
                        response = requests.post(
                            f"{api_url}/batch-predict", 
                            files={"file": (uploaded_file.name, uploaded_file.getvalue())}
                        )
                        result = response.json()
                        if response.status_code == 200:
                            st.session_state.batch_results = result["results"]
                            st.session_state.analysis_complete = True
                            st.session_state.uploaded_filename = uploaded_file.name
                            st.rerun()
                        else:
                            st.error(f"❌ Analysis Failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"🔌 Processing Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ File Error: {str(e)}")

    # Show analysis results AFTER EXECUTION
    if st.session_state.analysis_complete and st.session_state.batch_results:
        df = pd.DataFrame(st.session_state.batch_results)
        df['Risk_Level'] = df['probability'].apply(lambda x:
            '🔴 High Risk' if x > 0.7 else '🟡 Medium Risk' if x > 0.4 else '🟢 Low Risk')
        df['Churn_Prediction'] = df['prediction'].map({0: '✅ Retained', 1: '⚠️ Churn'})
        df['Probability_Percent'] = (df['probability'] * 100).round(1).astype(str) + '%'

        st.markdown("### 📋 Analysis Results")
        total = len(df)
        high_risk = len(df[df['Risk_Level'] == '🔴 High Risk'])
        medium_risk = len(df[df['Risk_Level'] == '🟡 Medium Risk'])
        low_risk = len(df[df['Risk_Level'] == '🟢 Low Risk'])
        churned = df["prediction"].sum()
        churn_rate = (churned / total) * 100
        avg_probability = df["probability"].mean() * 100
        st.markdown("#### 📈 Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Customers", f"{total:,}")
        with col2: st.metric("High Risk", f"{high_risk:,}", f"{(high_risk/total*100):.1f}%")
        with col3: st.metric("Predicted Churn", f"{churned:,}", f"{churn_rate:.1f}%")
        with col4: st.metric("Avg Risk Score", f"{avg_probability:.1f}%")
        st.markdown("#### 📊 Detailed Customer Analysis")
        col1, col2 = st.columns([1, 3])
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level:",
                ["All", "🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
                key="risk_filter"
            )
        with col2:
            search_term = st.text_input("Search in results:", key="search_results")

        filtered_df = df.copy()
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
        if search_term:
            filtered_df = filtered_df[
                filtered_df.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
            ]
        display_columns = ['Churn_Prediction', 'Probability_Percent', 'Risk_Level']
        st.dataframe(
            filtered_df[display_columns].reset_index(drop=True),
            use_container_width=True,
            height=400
        )

        # Action buttons below result
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            csv_data = df.to_csv(index=False)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "📥 Download Analysis",
                data=csv_data,
                file_name=f"churn_analysis_report_{timestamp}.csv",
                mime="text/csv",
                key="download_detailed_report",
                use_container_width=True
            )
        with col3:
            if st.button("📊 View Dashboard", key="view_dashboard_btn", use_container_width=True):
                st.session_state.dashboard_data = {
                    'total_customers': total,
                    'churned': churned,
                    'churn_rate': churn_rate,
                    'avg_probability': avg_probability,
                    'risk_distribution': df['Risk_Level'].value_counts().to_dict(),
                    'probability_data': df['probability'].tolist()
                }
                st.query_params["dashboard"]="true"
                st.query_params["ts"]= timestamp
                st.rerun()
        # with col1:
        #     if st.button("🔄 Analyze New File", key="new_analysis"):
        #         st.session_state.analysis_complete = False
        #         st.session_state.batch_results = None
        #         st.session_state.dashboard_data = None
        #         st.query_params()
        #         st.rerun()
