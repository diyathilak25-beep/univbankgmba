import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, preprocess_data, filter_data
from utils.analytics import DescriptiveAnalytics, DiagnosticAnalytics
from utils.visualizations import Visualizer
from models.predictor import LoanPredictor

# Page configuration
st.set_page_config(
    page_title="Universal Bank - Personal Loan Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #e8f8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏦 Universal Bank - Personal Loan Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Understanding Customer Behavior for Personal Loan Acceptance</p>', unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def get_data():
    df = load_data("UniversalBank.csv")
    if df is not None:
        df = preprocess_data(df)
    return df

df = get_data()

if df is None:
    st.error("Please upload the UniversalBank.csv file to proceed.")
    uploaded_file = st.file_uploader("Upload UniversalBank.csv", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        st.success("File uploaded successfully!")
        st.rerun()
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

income_min, income_max = int(df['Income'].min()), int(df['Income'].max())
income_range = st.sidebar.slider(
    "Income Range ($K)",
    min_value=income_min,
    max_value=income_max,
    value=(income_min, income_max)
)

education_options = {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced/Professional'}
education_selected = st.sidebar.multiselect(
    "Education Level",
    options=list(education_options.keys()),
    format_func=lambda x: education_options[x],
    default=list(education_options.keys())
)

family_sizes = sorted(df['Family'].unique())
family_selected = st.sidebar.multiselect(
    "Family Size",
    options=family_sizes,
    default=family_sizes
)

filtered_df = filter_data(df, income_range, education_selected, family_selected)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Customers:** {len(filtered_df):,} / {len(df):,}")

# Initialize analytics and visualizer
desc_analytics = DescriptiveAnalytics(filtered_df)
diag_analytics = DiagnosticAnalytics(filtered_df)
visualizer = Visualizer(filtered_df)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptive Analytics", 
    "🔬 Diagnostic Analytics", 
    "🎯 Predictive Analytics",
    "💡 Prescriptive Analytics",
    "🎁 Personalized Offers"
])

# TAB 1: DESCRIPTIVE ANALYTICS
with tab1:
    st.header("📊 Descriptive Analytics")
    st.markdown("*Understanding the current state of customer data and loan acceptance patterns*")
    
    stats = desc_analytics.get_basic_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{stats['total_customers']:,}")
    with col2:
        st.metric("Loans Accepted", f"{stats['loan_accepted']:,}")
    with col3:
        st.metric("Loans Rejected", f"{stats['loan_rejected']:,}")
    with col4:
        st.metric("Acceptance Rate", f"{stats['acceptance_rate']:.1f}%")
    
    st.markdown("---")
    
    st.subheader("📈 Demographic Summary")
    
    demo_summary = desc_analytics.get_demographic_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Age Statistics**")
        st.write(f"- Mean: {demo_summary['age']['mean']:.1f} years")
        st.write(f"- Median: {demo_summary['age']['median']:.1f} years")
        st.write(f"- Range: {demo_summary['age']['min']} - {demo_summary['age']['max']} years")
    
    with col2:
        st.markdown("**Income Statistics**")
        st.write(f"- Mean: ${demo_summary['income']['mean']:.1f}K")
        st.write(f"- Median: ${demo_summary['income']['median']:.1f}K")
        st.write(f"- Range: ${demo_summary['income']['min']}K - ${demo_summary['income']['max']}K")
    
    with col3:
        st.markdown("**Credit Card Spending**")
        st.write(f"- Mean: ${demo_summary['ccavg']['mean']:.2f}K/month")
        st.write(f"- Median: ${demo_summary['ccavg']['median']:.2f}K/month")
        st.write(f"- Has Mortgage: {demo_summary['mortgage']['has_mortgage_pct']:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(visualizer.plot_loan_acceptance_donut(), use_container_width=True)
    
    with col2:
        st.plotly_chart(visualizer.plot_education_acceptance(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(visualizer.plot_age_distribution(), use_container_width=True)
    
    with col2:
        st.plotly_chart(visualizer.plot_income_distribution(), use_container_width=True)
    
    st.subheader("📊 Average Metrics by Loan Status")
    avg_metrics = desc_analytics.get_average_metrics_by_loan_status()
    avg_metrics.index = ['Rejected', 'Accepted']
    st.dataframe(avg_metrics.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>🔍 Key Insights from Descriptive Analytics:</strong>
    <ul>
        <li>The overall loan acceptance rate is relatively low, indicating a selective customer base for personal loans.</li>
        <li>Higher education levels show correlation with loan acceptance.</li>
        <li>Income distribution varies significantly between accepted and rejected customers.</li>
        <li>Age distribution is fairly uniform across both groups.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 2: DIAGNOSTIC ANALYTICS
with tab2:
    st.header("🔬 Diagnostic Analytics")
    st.markdown("*Understanding WHY customers accept or reject personal loans*")
    
    st.subheader("📊 Accepted vs Rejected: Key Differences")
    
    comparison_df = diag_analytics.compare_groups()
    
    st.dataframe(
        comparison_df.style.format({
            'Loan Accepted': '{:.2f}',
            'Loan Rejected': '{:.2f}',
            'Difference': '{:.2f}',
            '% Difference': '{:.1f}%'
        }).background_gradient(subset=['% Difference'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    st.subheader("🎯 Key Drivers of Loan Acceptance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        driver_df = diag_analytics.get_key_drivers()
        st.plotly_chart(visualizer.plot_key_drivers(driver_df), use_container_width=True)
    
    with col2:
        st.markdown("**Top Influencing Factors:**")
        for idx, row in driver_df.head(5).iterrows():
            impact_icon = "📈" if row['Impact'] == 'Positive' else "📉"
            st.write(f"{impact_icon} **{row['Feature']}**: {row['Correlation']:.3f}")
    
    st.markdown("---")
    
    st.subheader("🔥 Feature Correlation Heatmap")
    corr_matrix = diag_analytics.get_correlation_matrix()
    st.plotly_chart(visualizer.plot_correlation_heatmap(corr_matrix), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🏦 Banking Services Impact on Loan Acceptance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        services_df = diag_analytics.analyze_banking_services()
        st.plotly_chart(visualizer.plot_banking_services_analysis(services_df), use_container_width=True)
    
    with col2:
        st.markdown("**Service Impact Analysis:**")
        for idx, row in services_df.iterrows():
            lift = row['Lift']
            if lift > 1.5:
                st.success(f"✅ **{row['Service']}**: {lift:.1f}x more likely to accept")
            elif lift > 1:
                st.info(f"ℹ️ **{row['Service']}**: {lift:.1f}x more likely to accept")
            else:
                st.warning(f"⚠️ **{row['Service']}**: No significant impact")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(visualizer.plot_income_vs_loan(), use_container_width=True)
    
    with col2:
        st.plotly_chart(visualizer.plot_ccavg_vs_loan(), use_container_width=True)
    
    st.plotly_chart(visualizer.plot_scatter_income_ccavg(), use_container_width=True)
    
    st.plotly_chart(visualizer.plot_family_size_analysis(), use_container_width=True)
    
    st.subheader("👥 Customer Segment Analysis")
    segment_df = diag_analytics.segment_analysis()
    st.plotly_chart(visualizer.plot_segment_analysis(segment_df), use_container_width=True)
    
    st.subheader("🔄 Interactive Drill-Down Analysis")
    st.markdown("*Click on segments to drill down into Education → Income Group → Loan Status*")
    st.plotly_chart(visualizer.plot_interactive_drilldown(), use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>🔍 Key Diagnostic Insights:</strong>
    <ul>
        <li><strong>Income is the strongest predictor</strong> - Customers who accepted loans have significantly higher average income.</li>
        <li><strong>CD Account holders are much more likely to accept</strong> - Having a CD account shows strong positive correlation with loan acceptance.</li>
        <li><strong>Higher education correlates with acceptance</strong> - Advanced/Professional degree holders show higher acceptance rates.</li>
        <li><strong>Credit card spending matters</strong> - Higher CC spending indicates higher loan acceptance probability.</li>
        <li><strong>Family size has moderate impact</strong> - Larger families show slightly higher acceptance rates.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 3: PREDICTIVE ANALYTICS
with tab3:
    st.header("🎯 Predictive Analytics")
    st.markdown("*Building models to predict loan acceptance probability*")
    
    model_type = st.selectbox(
        "Select Model Type",
        options=['random_forest', 'gradient_boosting', 'logistic_regression'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    predictor = LoanPredictor(filtered_df)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            metrics, importance, features = predictor.train_model(model_type)
            st.session_state['model_trained'] = True
            st.session_state['metrics'] = metrics
            st.session_state['importance'] = importance
            st.session_state['features'] = features
            st.session_state['predictor'] = predictor
    
    if st.session_state.get('model_trained', False):
        metrics = st.session_state['metrics']
        importance = st.session_state['importance']
        
        st.success("✅ Model trained successfully!")
        
        st.subheader("📈 Model Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision']*100:.1f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall']*100:.1f}%")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']*100:.1f}%")
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']*100:.1f}%")
        
        st.markdown("---")
        
        st.subheader("🎯 Feature Importance")
        
        import plotly.express as px
        fig = px.bar(
            importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Predicting Loan Acceptance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("🧑‍💼 Predict for Individual Customer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=35)
            experience = st.number_input("Experience (years)", min_value=0, max_value=50, value=10)
            income = st.number_input("Income ($K)", min_value=0, max_value=300, value=80)
            family = st.selectbox("Family Size", options=[1, 2, 3, 4])
        
        with col2:
            ccavg = st.number_input("CC Avg Spending ($K)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
            education = st.selectbox("Education", options=[1, 2, 3], format_func=lambda x: {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced/Professional'}[x])
            mortgage = st.number_input("Mortgage ($K)", min_value=0, max_value=700, value=0)
        
        with col3:
            securities = st.selectbox("Securities Account", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            cd_account = st.selectbox("CD Account", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            online = st.selectbox("Online Banking", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            creditcard = st.selectbox("Credit Card", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        
        if st.button("🔮 Predict Loan Acceptance"):
            customer_data = {
                'Age': age,
                'Experience': experience,
                'Income': income,
                'Family': family,
                'CCAvg': ccavg,
                'Education': education,
                'Mortgage': mortgage,
                'Securities_Account': securities,
                'CD_Account': cd_account,
                'Online': online,
                'CreditCard': creditcard
            }
            
            predictor = st.session_state['predictor']
            features = st.session_state['features']
            prediction, probability = predictor.predict_single(customer_data, features)
            
            if prediction is not None:
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success(f"✅ **Prediction: LIKELY TO ACCEPT**")
                    else:
                        st.error(f"❌ **Prediction: UNLIKELY TO ACCEPT**")
                
                with col2:
                    st.metric("Acceptance Probability", f"{probability*100:.1f}%")
                
                import plotly.graph_objects as go
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Loan Acceptance Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71" if probability >= 0.5 else "#e74c3c"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ffebee"},
                            {'range': [30, 60], 'color': "#fff3e0"},
                            {'range': [60, 100], 'color': "#e8f5e9"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

# TAB 4: PRESCRIPTIVE ANALYTICS
with tab4:
    st.header("💡 Prescriptive Analytics")
    st.markdown("*Actionable recommendations for targeting customers*")
    
    st.subheader("🎯 Target Segment Recommendations")
    
    segment_df = diag_analytics.segment_analysis()
    segment_df_sorted = segment_df.sort_values('Acceptance Rate', ascending=False)
    
    st.markdown("""
    <div class="recommendation-box">
    <strong>📋 Priority Targeting Recommendations:</strong>
    </div>
    """, unsafe_allow_html=True)
    
    for idx, row in segment_df_sorted.iterrows():
        if row['Acceptance Rate'] > 20:
            priority = "🔴 HIGH PRIORITY"
        elif row['Acceptance Rate'] > 10:
            priority = "🟡 MEDIUM PRIORITY"
        else:
            priority = "🟢 LOW PRIORITY"
        
        st.markdown(f"""
        **{priority}**: {row['Segment']}
        - Acceptance Rate: {row['Acceptance Rate']:.1f}%
        - Customer Count: {row['Count']:,}
        """)
    
    st.markdown("---")
    
    st.subheader("📝 Detailed Marketing Recommendations")
    
    recommendations = [
        {
            "segment": "High Income + CD Account Holders",
            "description": "Customers with income >$100K and existing CD accounts",
            "strategy": "Premium personal loan packages with competitive rates",
            "expected_conversion": "25-35%",
            "channel": "Relationship Manager outreach, Email campaigns"
        },
        {
            "segment": "Graduate/Advanced Education",
            "description": "Customers with Graduate or Advanced/Professional degrees",
            "strategy": "Education-focused loan products, professional development loans",
            "expected_conversion": "15-25%",
            "channel": "LinkedIn ads, Professional network partnerships"
        },
        {
            "segment": "High CC Spenders",
            "description": "Customers with CC spending >$3K/month",
            "strategy": "Debt consolidation offers, Balance transfer promotions",
            "expected_conversion": "18-28%",
            "channel": "In-app notifications, Credit card statement inserts"
        },
        {
            "segment": "Large Families with Mortgage",
            "description": "Family size ≥3 with existing mortgage",
            "strategy": "Home improvement loans, Family emergency funds",
            "expected_conversion": "12-20%",
            "channel": "Direct mail, Branch promotions"
        },
        {
            "segment": "Online Banking Users",
            "description": "Active online banking customers",
            "strategy": "Quick digital loan applications, Instant approval offers",
            "expected_conversion": "10-15%",
            "channel": "Online banking portal, Mobile app push notifications"
        }
    ]
    
    for rec in recommendations:
        with st.expander(f"📌 {rec['segment']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Description:** {rec['description']}")
                st.markdown(f"**Strategy:** {rec['strategy']}")
            with col2:
                st.markdown(f"**Expected Conversion:** {rec['expected_conversion']}")
                st.markdown(f"**Recommended Channels:** {rec['channel']}")
    
    st.markdown("---")
    
    st.subheader("🚀 Campaign Optimization Suggestions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="recommendation-box">
        <strong>✅ DO:</strong>
        <ul>
            <li>Focus on high-income customers ($100K+)</li>
            <li>Prioritize CD account holders</li>
            <li>Target customers with higher education</li>
            <li>Offer competitive rates to high CC spenders</li>
            <li>Use digital channels for online banking users</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ AVOID:</strong>
        <ul>
            <li>Mass marketing to low-income segments</li>
            <li>Ignoring existing banking relationships</li>
            <li>One-size-fits-all loan products</li>
            <li>Overlooking family financial needs</li>
            <li>Neglecting digital-first customers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("💰 Estimated Campaign ROI")
    
    total_customers = len(filtered_df)
    high_potential = len(filtered_df[(filtered_df['Income'] > 100) | (filtered_df['CD_Account'] == 1)])
    avg_loan_value = 50
    conversion_rate = 0.25
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Potential Customers", f"{high_potential:,}")
    with col2:
        st.metric("Expected Conversions", f"{int(high_potential * conversion_rate):,}")
    with col3:
        st.metric("Potential Loan Volume", f"${int(high_potential * conversion_rate * avg_loan_value):,}K")

# TAB 5: PERSONALIZED OFFERS
with tab5:
    st.header("🎁 Personalized Offers Generator")
    st.markdown("*Generate personalized loan offers for customers predicted to be interested*")
    
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train a model in the Predictive Analytics tab first.")
    else:
        predictor = st.session_state['predictor']
        features = st.session_state['features']
        
        st.subheader("🔧 Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_probability = st.slider(
                "Minimum Interest Probability",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Only generate offers for customers with probability above this threshold"
            )
        
        with col2:
            max_customers = st.number_input(
                "Maximum Customers to Process",
                min_value=10,
                max_value=len(filtered_df),
                value=min(100, len(filtered_df)),
                help="Limit the number of customers to process"
            )
        
        if st.button("🎯 Generate Personalized Offers", type="primary"):
            with st.spinner("Generating personalized offers..."):
                sample_df = filtered_df.head(max_customers).copy()
                result_df = predictor.generate_personalized_offers(sample_df, features)
                
                if result_df is not None:
                    interested_df = result_df[result_df['Interest_Probability'] >= min_probability].copy()
                    
                    st.success(f"✅ Generated offers for {len(interested_df)} interested customers out of {len(sample_df)} processed")
                    
                    st.markdown("---")
                    
                    st.subheader("📊 Offer Generation Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Customers Processed", len(sample_df))
                    with col2:
                        st.metric("Predicted Interested", len(interested_df))
                    with col3:
                        st.metric("Interest Rate", f"{len(interested_df)/len(sample_df)*100:.1f}%")
                    with col4:
                        avg_prob = interested_df['Interest_Probability'].mean() if len(interested_df) > 0 else 0
                        st.metric("Avg Probability", f"{avg_prob*100:.1f}%")
                    
                    st.markdown("---")
                    
                    st.subheader("🎁 Generated Personalized Offers")
                    
                    if len(interested_df) > 0:
                        high_priority = interested_df[interested_df['Interest_Probability'] >= 0.8]
                        medium_priority = interested_df[(interested_df['Interest_Probability'] >= 0.6) & (interested_df['Interest_Probability'] < 0.8)]
                        standard = interested_df[interested_df['Interest_Probability'] < 0.6]
                        
                        offer_tab1, offer_tab2, offer_tab3 = st.tabs([
                            f"🔴 High Priority ({len(high_priority)})",
                            f"🟡 Medium Priority ({len(medium_priority)})",
                            f"🟢 Standard ({len(standard)})"
                        ])
                        
                        def display_offers(df, priority_label):
                            if len(df) == 0:
                                st.info(f"No {priority_label} customers found.")
                                return
                            
                            for idx, row in df.iterrows():
                                offer = row['Personalized_Offer']
                                if offer is not None:
                                    with st.expander(f"Customer #{idx} - Probability: {row['Interest_Probability']*100:.1f}%"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown("**Customer Profile:**")
                                            st.write(f"- Age: {row['Age']}")
                                            st.write(f"- Income: ${row['Income']}K")
                                            st.write(f"- Education: {row.get('Education_Label', row['Education'])}")
                                            st.write(f"- Family Size: {row['Family']}")
                                            st.write(f"- CC Spending: ${row['CCAvg']}K/month")
                                        
                                        with col2:
                                            st.markdown("**Personalized Offer:**")
                                            st.write(f"- 💰 Recommended Loan: {offer['recommended_loan_amount']}")
                                            st.write(f"- 📊 Interest Rate: {offer['interest_rate']}")
                                            st.write(f"- ⭐ Priority: {offer['priority_level']}")
                                            st.write(f"- 🎯 Score: {offer['probability_score']}")
                                        
                                        st.markdown("**Special Offers:**")
                                        for special in offer['special_offers']:
                                            st.write(f"  ✨ {special}")
                        
                        with offer_tab1:
                            display_offers(high_priority, "high priority")
                        
                        with offer_tab2:
                            display_offers(medium_priority, "medium priority")
                        
                        with offer_tab3:
                            display_offers(standard, "standard")
                        
                        st.markdown("---")
                        
                        st.subheader("📥 Export Offers")
                        
                        export_df = interested_df[['Age', 'Income', 'Education', 'Family', 'CCAvg', 
                                                   'Interest_Probability', 'Personalized_Offer']].copy()
                        
                        export_df['Recommended_Loan'] = export_df['Personalized_Offer'].apply(
                            lambda x: x['recommended_loan_amount'] if x else None
                        )
                        export_df['Interest_Rate'] = export_df['Personalized_Offer'].apply(
                            lambda x: x['interest_rate'] if x else None
                        )
                        export_df['Priority_Level'] = export_df['Personalized_Offer'].apply(
                            lambda x: x['priority_level'] if x else None
                        )
                        export_df['Special_Offers'] = export_df['Personalized_Offer'].apply(
                            lambda x: '; '.join(x['special_offers']) if x else None
                        )
                        
                        export_df = export_df.drop('Personalized_Offer', axis=1)
                        
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Offers as CSV",
                            data=csv,
                            file_name="personalized_loan_offers.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No customers meet the minimum probability threshold. Try lowering the threshold.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏦 Universal Bank Personal Loan Analysis Dashboard</p>
    <p>Built with Streamlit | Data Analytics for Better Business Decisions</p>
</div>
""", unsafe_allow_html=True)
