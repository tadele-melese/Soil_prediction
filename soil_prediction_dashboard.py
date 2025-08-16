import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore, gaussian_kde
import plotly.figure_factory as ff
try:
    import kaleido
except ImportError:
    kaleido = None

# Set page configuration
st.set_page_config(
    page_title="🌍 #1 Vibrant Soil Prediction Dashboard 🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    .main {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%) !important;
        transition: all 0.3s ease-in-out;
        font-family: 'Poppins', sans-serif;
    }
    .stSidebar {
        background: linear-gradient(135deg, #ff9a8b 0%, #ff6a88 100%) !important;
        border-right: 4px solid #ff4500 !important;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #ff4500 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton > button {
        background: linear-gradient(45deg, #32cd32, #228b22) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    .stSlider > div {
        background-color: #ffd700 !important;
    }
    .block-container {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.9) !important;
        transition: transform 0.3s ease;
    }
    .block-container:hover {
        transform: translateY(-5px);
    }
    .stMultiSelect > div, .stSelectbox > div {
        background: #fffacd !important;
        border: 3px solid #ff4500 !important;
        border-radius: 10px;
        transition: border-color 0.3s;
    }
    .stMultiSelect > div:hover, .stSelectbox > div:hover {
        border-color: #ff6347 !important;
    }
    .stSidebar .uploadedFile {
        border: 3px dashed #ff4500 !important;
        padding: 15px;
        border-radius: 15px;
        background: #fff5ee !important;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #333;
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 130%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌱 #1 Vibrant Soil Prediction Dashboard 🌍")
st.markdown("""
Welcome to the **most vibrant and powerful** soil analysis tool! Explore soil types (Sandy, Loamy, Clayey) with stunning charts, predict soil types, and get crop recommendations for Amhara farming! 🚜
<div class='tooltip'>What is this dashboard?<span class='tooltiptext'>This tool analyzes soil properties (e.g., sand, clay, pH) to predict soil types and suggest crops like teff. Perfect for beginners and farmers!</span></div>
""", unsafe_allow_html=True)

# Load data with progress bar
@st.cache_data
def load_data():
    progress = st.progress(0)
    try:
        df = pd.read_csv('simulated_soil_data.csv')
        progress.progress(20)
        st.write(f"Initial dataset size: {len(df)} rows")
        df = df.dropna()
        progress.progress(40)
        df['Soil_Type'] = df['Soil_Type'].replace('Loamyy', 'Loamy')
        df = df[df['Soil_pH'] > 0]
        df = df[df['Soil_pH'].between(4, 9)]
        df = df[df['Organic_Carbon'].between(0, 10)]
        df = df[df['Clay_Content'].between(0, 100) & 
                df['Sand_Content'].between(0, 100) & 
                df['Silt_Content'].between(0, 100)]
        df = df[df['EC'].between(0, 5)]
        progress.progress(80)
        st.write(f"Dataset size after range filtering: {len(df)} rows")
        st.write("Class distribution after cleaning:", df['Soil_Type'].value_counts().to_dict())
        progress.progress(100)
        if df.isnull().any().any():
            st.warning("Warning: Dataset contains missing values after cleaning!")
        return df
    except FileNotFoundError:
        st.error("simulated_soil_data.csv not found. Ensure the file is in the same directory.")
        progress.progress(100)
        return None

df = load_data()

# Sidebar
if df is not None:
    all_features = df.select_dtypes(include=np.number).columns.tolist()
    with st.sidebar:
        st.header("🛠️ Dashboard Controls")
        theme = st.selectbox("🎨 Choose Theme", ["Vibrant", "Dark", "Light", "Pastel"], help="Change the dashboard's look!")
        st.image(f"https://via.placeholder.com/50/{'ff4500' if theme=='Vibrant' else '2f4f4f' if theme=='Dark' else 'f0f0f0' if theme=='Light' else 'ffe4e1'}/ffffff?text={theme[0]}", caption=f"{theme} Theme Preview")
        if theme == "Dark":
            st.markdown("""
            <style>
                .main {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important; color: white !important;}
                .block-container {background: #2f4f4f !important;}
                h1, h2, h3 {color: #ffd700 !important;}
            </style>
            """, unsafe_allow_html=True)
            plotly_template = "plotly_dark"
            st.success("Switched to Dark theme! 🌙")
        elif theme == "Light":
            st.markdown("""
            <style>
                .main {background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%) !important;}
                .block-container {background: #f5f5f5 !important;}
            </style>
            """, unsafe_allow_html=True)
            plotly_template = "plotly_white"
            st.success("Switched to Light theme! ☀️")
        elif theme == "Pastel":
            st.markdown("""
            <style>
                .main {background: linear-gradient(135deg, #ffb6c1 0%, #b0e0e6 100%) !important;}
                .block-container {background: #fff0f5 !important;}
            </style>
            """, unsafe_allow_html=True)
            plotly_template = "plotly"
            st.success("Switched to Pastel theme! 🌸")
        else:
            plotly_template = "plotly"
            st.success("Switched to Vibrant theme! 🔥")
        
        st.header("📊 Data Filters")
        soil_types = st.multiselect("Filter by Soil Type", df['Soil_Type'].unique(), default=df['Soil_Type'].unique())
        ph_range = st.slider("Soil pH Range", 4.0, 9.0, (4.0, 9.0))
        clay_range = st.slider("Clay Content Range", 0.0, 100.0, (0.0, 100.0))
        sand_range = st.slider("Sand Content Range", 0.0, 100.0, (0.0, 100.0))
        silt_range = st.slider("Silt Content Range", 0.0, 100.0, (0.0, 100.0))
        organic_range = st.slider("Organic Carbon Range", 0.0, 10.0, (0.0, 10.0))
        ec_range = st.slider("EC Range", 0.0, 5.0, (0.0, 5.0))
        filtered_df = df[df['Soil_Type'].isin(soil_types) & df['Soil_pH'].between(ph_range[0], ph_range[1]) &
                         df['Clay_Content'].between(clay_range[0], clay_range[1]) &
                         df['Sand_Content'].between(sand_range[0], sand_range[1]) &
                         df['Silt_Content'].between(silt_range[0], silt_range[1]) &
                         df['Organic_Carbon'].between(organic_range[0], organic_range[1]) &
                         df['EC'].between(ec_range[0], ec_range[1])]
        st.write(f"Filtered dataset size: {len(filtered_df)} rows")
        
        st.header("📈 EDA Features")
        suggested_features = df[all_features].var().sort_values(ascending=False).index[:3].tolist()
        st.write(f"Suggested features (high variance): {', '.join(suggested_features)}")
        selected_features = st.multiselect(
            "Select Features for EDA", all_features, default=suggested_features,
            help="Choose 2+ features for most plots."
        )
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features for interactive plots.")
        
        st.header("📂 Upload CSV")
        uploaded_file = st.file_uploader("Upload your own CSV", type=["csv"], help="CSV must have Soil_pH, Organic_Carbon, etc.")
        if uploaded_file is not None:
            custom_df = pd.read_csv(uploaded_file)
            custom_df['Soil_Type'] = custom_df['Soil_Type'].replace('Loamyy', 'Loamy')
            custom_df = custom_df.dropna()
            custom_df = custom_df[custom_df['Soil_pH'].between(4, 9)]
            custom_df = custom_df[custom_df['Organic_Carbon'].between(0, 10)]
            custom_df = custom_df[custom_df['Clay_Content'].between(0, 100) & 
                                 custom_df['Sand_Content'].between(0, 100) & 
                                 custom_df['Silt_Content'].between(0, 100)]
            custom_df = custom_df[custom_df['EC'].between(0, 5)]
            filtered_df = custom_df[custom_df['Soil_Type'].isin(soil_types) & custom_df['Soil_pH'].between(ph_range[0], ph_range[1])]
            st.success(f"Uploaded {len(custom_df)} rows. Filtered to {len(filtered_df)} rows.")
            all_features = filtered_df.select_dtypes(include=np.number).columns.tolist()
        
        st.header("⚙️ Model Settings")
        n_estimators = st.slider("Number of Trees", 50, 200, 100, help="More trees = more accurate but slower model")
        
        st.header("📥 Export Options")
        if st.button("Download EDA Report"):
            report = "EDA Report\n" + filtered_df.describe().to_string()
            st.download_button("Download Report TXT", report.encode(), "eda_report.txt")
        
        st.header("🚀 Quick Start")
        if st.button("Load Default Settings"):
            st.session_state.selected_features = suggested_features
            st.session_state.soil_types = df['Soil_Type'].unique().tolist()
            st.session_state.ph_range = (4.0, 9.0)
            st.success("Loaded default settings! Try the tabs to explore.")

# Feature columns
feature_columns = ['Soil_pH', 'Organic_Carbon', 'Clay_Content', 'Sand_Content', 'Silt_Content', 
                   'EC', 'Clay_Sand_Ratio', 'Texture_Sum', 'pH_EC_Interaction', 'Organic_Texture_Ratio']

# Load or train model
@st.cache_resource
def get_model(n_estimators=100):
    if df is None:
        return None, None, None
    try:
        model = joblib.load('improved_models/model_random_forest_(smote).pkl')
        scaler = joblib.load('improved_models/scaler.pkl')
        label_encoder = joblib.load('improved_models/label_encoder.pkl')
        # Check if saved model's features match feature_columns
        if hasattr(scaler, 'feature_names_in_') and list(scaler.feature_names_in_) != feature_columns:
            st.warning(f"Saved model uses {scaler.feature_names_in_}, but expected {feature_columns}. Forcing retraining...")
            raise FileNotFoundError("Feature mismatch")
        st.success("Loaded pre-trained model! 🚀")
        return model, scaler, label_encoder
    except (FileNotFoundError, AttributeError):
        st.warning("Training a new model...")
        data = df.copy()
        data['Clay_Sand_Ratio'] = data['Clay_Content'] / (data['Sand_Content'] + 1e-6)
        data['Texture_Sum'] = data['Clay_Content'] + data['Sand_Content'] + data['Silt_Content']
        data['pH_EC_Interaction'] = data['Soil_pH'] * data['EC']
        data['Organic_Texture_Ratio'] = data['Organic_Carbon'] / (data['Texture_Sum'] + 1e-6)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['Soil_Type'])
        X = data[feature_columns]
        class_counts = pd.Series(y).value_counts()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        scaler.feature_names_in_ = X.columns.tolist()
        min_train_samples = min(pd.Series(y_train).value_counts())
        k_neighbors = min(5, max(1, min_train_samples - 1))
        if k_neighbors < 1 or min_train_samples < 2:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train_scaled, y_train)
        else:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train_resampled, y_train_resampled)
        os.makedirs('improved_models', exist_ok=True)
        joblib.dump(model, 'improved_models/model_random_forest_(smote).pkl')
        joblib.dump(scaler, 'improved_models/scaler.pkl')
        joblib.dump(label_encoder, 'improved_models/label_encoder.pkl')
        st.success("Trained and saved new model with features: {}".format(feature_columns))
        return model, scaler, label_encoder
model, scaler, label_encoder = get_model(n_estimators)

# Crop recommendation
def recommend_crops(soil_type, soil_ph):
    recommendations = {
        'Sandy': {'pH_range': (6.0, 7.5), 'crops': ['Teff', 'Sorghum', 'Carrots'], 'advice': 'Sandy soils drain quickly. Add organic matter.'},
        'Loamy': {'pH_range': (6.0, 7.0), 'crops': ['Teff', 'Wheat', 'Maize'], 'advice': 'Loamy soils are ideal. Maintain pH.'},
        'Clayey': {'pH_range': (6.5, 7.5), 'crops': ['Rice', 'Barley', 'Legumes'], 'advice': 'Clayey soils hold water. Improve drainage.'}
    }
    rec = recommendations.get(soil_type, {'crops': [], 'advice': 'Unknown soil type'})
    ph_ok = rec['pH_range'][0] <= soil_ph <= rec['pH_range'][1]
    status = "✅ Good pH" if ph_ok else f"⚠️ Adjust pH (ideal: {rec['pH_range'][0]}–{rec['pH_range'][1]})"
    return rec['crops'], rec['advice'], status

if filtered_df is not None and model is not None:
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Data Overview", "Basic EDA", "Advanced EDA", "Interactive Plots", 
        "Prediction", "Model Explanation", "Feature Importance", "Model Performance"
    ])

    with tab1:
        st.subheader("📊 Sample Data")
        st.markdown("<div class='tooltip'>What is this?<span class='tooltiptext'>Preview your soil data (pH, sand, clay, etc.).</span></div>", unsafe_allow_html=True)
        st.data_editor(filtered_df.head(20), use_container_width=True, column_config={
            "Soil_pH": st.column_config.NumberColumn("Soil pH", min_value=4, max_value=9, format="%.2f"),
            "Organic_Carbon": st.column_config.NumberColumn("Organic Carbon (%)", min_value=0, max_value=10, format="%.2f"),
            "Clay_Content": st.column_config.NumberColumn("Clay Content (%)", min_value=0, max_value=100, format="%.2f"),
            "Sand_Content": st.column_config.NumberColumn("Sand Content (%)", min_value=0, max_value=100, format="%.2f"),
            "Silt_Content": st.column_config.NumberColumn("Silt Content (%)", min_value=0, max_value=100, format="%.2f"),
            "EC": st.column_config.NumberColumn("EC (dS/m)", min_value=0, max_value=5, format="%.2f")
        })
        st.write(f"Total rows: {len(filtered_df)}")
        st.write("Soil Type Distribution:")
        st.dataframe(filtered_df['Soil_Type'].value_counts().reset_index(name='Count'), use_container_width=True)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV 📥",
            data=csv,
            file_name="soil_data.csv",
            mime="text/csv",
            help="Download the filtered dataset."
        )

    with tab2:
        st.subheader("🔍 Basic Exploratory Data Analysis")
        st.markdown("<div class='tooltip'>What is this?<span class='tooltiptext'>Simple charts to see soil type patterns.</span></div>", unsafe_allow_html=True)
        fig_type = px.histogram(
            filtered_df, x='Soil_Type', title="Distribution of Soil Types",
            color='Soil_Type', color_discrete_sequence=px.colors.qualitative.D3,
            template=plotly_template, animation_frame='Soil_Type'
        )
        st.plotly_chart(fig_type, use_container_width=True)
        if len(selected_features) >= 2:
            fig_scatter = px.scatter(
                filtered_df, x=selected_features[0], y=selected_features[1], color='Soil_Type',
                title=f"{selected_features[0]} vs {selected_features[1]}",
                animation_frame='Soil_pH', color_discrete_sequence=px.colors.qualitative.Vivid,
                hover_data=['Organic_Carbon', 'EC'], size='Clay_Content',
                template=plotly_template
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        fig_box = px.box(
            filtered_df, x='Soil_Type', y=selected_features[0] if selected_features else 'Soil_pH',
            title="Box Plot by Soil Type", color='Soil_Type',
            color_discrete_sequence=px.colors.qualitative.Set1,
            template=plotly_template
        )
        st.plotly_chart(fig_box, use_container_width=True)
        corr = filtered_df[selected_features].corr() if selected_features else filtered_df.select_dtypes(include=np.number).corr()
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto", title="Correlation Heatmap",
            color_continuous_scale='RdYlBu', template=plotly_template
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        if kaleido:
            buf = BytesIO()
            try:
                fig_corr.write_image(buf, format="png")
                st.download_button("Download Correlation Heatmap", buf.getvalue(), "correlation_heatmap.png", "image/png")
            except Exception as e:
                st.warning(f"Cannot export heatmap: {str(e)}")
        else:
            st.warning("Install kaleido for image export: `pip install -U kaleido`")

    with tab3:
        st.subheader("🎨 Advanced EDA Visualizations")
        st.markdown("<div class='tooltip'>What is this?<span class='tooltiptext'>16 vibrant charts to explore soil data deeply!</span></div>", unsafe_allow_html=True)
        if filtered_df.empty:
            st.warning("No data available after applying filters. Please adjust sidebar filters (e.g., select more soil types or widen pH range).")
        else:
            if len(selected_features) >= 2:
                st.write("Pair Plot")
                fig_pair = px.scatter_matrix(
                    filtered_df, dimensions=selected_features, color='Soil_Type',
                    title="Pair Plot of Selected Features",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    template=plotly_template
                )
                st.plotly_chart(fig_pair, use_container_width=True)
                if kaleido:
                    buf = BytesIO()
                    try:
                        fig_pair.write_image(buf, format="png")
                        st.download_button("Download Pair Plot", buf.getvalue(), "pair_plot.png", "image/png")
                    except Exception:
                        st.warning("Cannot export pair plot.")
            else:
                st.warning("Select at least 2 features for pair plot.")
            
            violin_feature = st.selectbox("Select Feature for Violin Plot", selected_features or all_features)
            fig_violin = px.violin(
                filtered_df, y=violin_feature, x='Soil_Type', color='Soil_Type',
                box=True, points="all", title=f"{violin_feature} Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel1,
                template=plotly_template
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
            if len(selected_features) >= 3:
                st.write("3D Scatter Plot")
                fig_3d = px.scatter_3d(
                    filtered_df, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                    color='Soil_Type', size='Soil_pH', opacity=0.7,
                    title="3D Scatter Plot", color_discrete_sequence=px.colors.qualitative.G10,
                    hover_data=['Organic_Carbon', 'EC'], template=plotly_template
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            
            if len(selected_features) >= 2:
                st.write("Density Contour")
                fig_density = px.density_contour(
                    filtered_df, x=selected_features[0], y=selected_features[1], color='Soil_Type',
                    title=f"Density Contour: {selected_features[0]} vs {selected_features[1]}",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    template=plotly_template
                )
                st.plotly_chart(fig_density, use_container_width=True)
            
            st.write("Outlier Detection")
            df_outliers = filtered_df.copy()
            outlier_feature = selected_features[0] if selected_features else 'Soil_pH'
            df_outliers['Z_Score'] = zscore(df_outliers[outlier_feature])
            fig_outliers = px.scatter(
                df_outliers, x='Soil_Type', y=outlier_feature,
                color=(df_outliers['Z_Score'].abs() > 3), title="Outliers Highlighted",
                color_discrete_map={False: 'blue', True: 'red'},
                template=plotly_template
            )
            st.plotly_chart(fig_outliers, use_container_width=True)
            
            if len(selected_features) >= 2:
                st.write("PCA (2D Projection)")
                try:
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(MinMaxScaler().fit_transform(filtered_df[selected_features]))
                    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
                    pca_df['Soil_Type'] = filtered_df['Soil_Type']
                    fig_pca = px.scatter(
                        pca_df, x='PC1', y='PC2', color='Soil_Type',
                        title="PCA 2D Projection", color_discrete_sequence=px.colors.qualitative.T10,
                        template=plotly_template
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                except ValueError as e:
                    st.warning(f"Cannot generate PCA plot: {str(e)}. Try selecting different features or adjusting filters.")
            
            if len(selected_features) >= 2:
                st.write("Parallel Coordinates")
                fig_parallel = px.parallel_coordinates(
                    filtered_df, dimensions=selected_features, color_continuous_scale=px.colors.diverging.Tealrose,
                    labels={col: col.replace('_', ' ') for col in selected_features},
                    title="Feature Relationships by Soil Type",
                    color=filtered_df['Soil_Type'].astype('category').cat.codes,
                    template=plotly_template
                )
                st.plotly_chart(fig_parallel, use_container_width=True)
            
            st.write("Quantile-Based Distribution")
            quantile_feature = st.selectbox("Select Feature for Quantile Plot", selected_features or all_features)
            df_quantiles = filtered_df.copy()
            try:
                df_quantiles['Quantile'] = pd.qcut(df_quantiles[quantile_feature], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                fig_quantile = px.box(
                    df_quantiles, x='Soil_Type', y=quantile_feature, color='Quantile',
                    title=f"{quantile_feature} by Quantiles",
                    color_discrete_sequence=px.colors.qualitative.D3,
                    template=plotly_template
                )
                st.plotly_chart(fig_quantile, use_container_width=True)
            except ValueError:
                st.warning(f"Cannot create quantiles for {quantile_feature}. Try another feature.")
            
            st.write("Correlation Network")
            corr_matrix = filtered_df[selected_features].corr() if selected_features else filtered_df.select_dtypes(include=np.number).corr()
            fig_network = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='Viridis', showscale=True, text=corr_matrix.values.round(2),
                texttemplate="%{text}", textfont={"size": 10}
            ))
            fig_network.update_layout(title="Correlation Network", template=plotly_template)
            st.plotly_chart(fig_network, use_container_width=True)
            
            for col in selected_features or all_features:
                fig_hist = px.histogram(
                    filtered_df, x=col, color='Soil_Type', title=f"Histogram of {col}",
                    color_discrete_sequence=px.colors.qualitative.Safe, marginal="rug", nbins=30,
                    template=plotly_template
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            if len(selected_features) >= 3:
                st.write("Bubble Chart")
                fig_bubble = px.scatter(
                    filtered_df, x=selected_features[0], y=selected_features[1],
                    size=selected_features[2], color='Soil_Type',
                    title=f"Bubble Chart: {selected_features[0]} vs {selected_features[1]}",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    template=plotly_template
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
            
            st.write("Radar Chart")
            radar_df = filtered_df.groupby('Soil_Type')[all_features].mean().reset_index()
            fig_radar = px.line_polar(
                radar_df, r=selected_features[0] if selected_features else 'Soil_pH', theta='Soil_Type',
                line_close=True, title="Radar Chart of Average Feature by Soil Type",
                color_discrete_sequence=px.colors.qualitative.Set2,
                template=plotly_template
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.write("Boxen Plot")
            boxen_feature = st.selectbox("Select Feature for Boxen Plot", selected_features or all_features)
            fig_boxen = px.box(
                filtered_df, x='Soil_Type', y=boxen_feature, color='Soil_Type',
                title=f"Boxen Plot: {boxen_feature}", color_discrete_sequence=px.colors.qualitative.Set3,
                template=plotly_template, boxmode='group'
            )
            st.plotly_chart(fig_boxen, use_container_width=True)
            
            st.write("Strip Plot")
            strip_feature = st.selectbox("Select Feature for Strip Plot", selected_features or all_features)
            fig_strip = px.strip(
                filtered_df, x='Soil_Type', y=strip_feature, color='Soil_Type',
                title=f"Strip Plot: {strip_feature}", color_discrete_sequence=px.colors.qualitative.Dark24,
                template=plotly_template
            )
            st.plotly_chart(fig_strip, use_container_width=True)
            
            if len(selected_features) >= 2:
                st.write("Joint Plot")
                fig_joint = px.scatter(
                    filtered_df, x=selected_features[0], y=selected_features[1], 
                    marginal_x="histogram", marginal_y="histogram",
                    color='Soil_Type', title=f"Joint Plot: {selected_features[0]} vs {selected_features[1]}",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    template=plotly_template
                )
                st.plotly_chart(fig_joint, use_container_width=True)
            
            st.write("Ridgeline Plot")
            ridge_feature = st.selectbox("Select Feature for Ridgeline Plot", selected_features or all_features)
            fig_ridge = go.Figure()
            for soil_type in filtered_df['Soil_Type'].unique():
                data = filtered_df[filtered_df['Soil_Type'] == soil_type][ridge_feature]
                kde = gaussian_kde(data)
                x = np.linspace(data.min(), data.max(), 100)
                y = kde(x)
                fig_ridge.add_trace(go.Scatter(x=x, y=y, mode='lines', name=soil_type, fill='tozeroy'))
            fig_ridge.update_layout(title=f"Ridgeline Plot: {ridge_feature} by Soil Type", template=plotly_template)
            st.plotly_chart(fig_ridge, use_container_width=True)
            
            st.write("Sankey Diagram")
            sankey_feature = st.selectbox("Select Feature for Sankey Diagram", selected_features or all_features)
            sankey_df = filtered_df.groupby(['Soil_Type', pd.cut(filtered_df[sankey_feature], bins=4)]).size().reset_index(name='count')
            sankey_df['bin'] = sankey_df[sankey_feature].apply(lambda x: f"{x.left}-{x.right}")
            labels = list(filtered_df['Soil_Type'].unique()) + list(sankey_df['bin'].unique())
            source = []
            target = []
            value = []
            for i, row in sankey_df.iterrows():
                source_idx = labels.index(row['Soil_Type'])
                target_idx = labels.index(row['bin'])
                source.append(source_idx)
                target.append(target_idx)
                value.append(row['count'])
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(label=labels, color=px.colors.qualitative.D3),
                link=dict(source=source, target=target, value=value, color='rgba(255, 69, 0, 0.5)')
            )])
            fig_sankey.update_layout(title=f"Sankey Diagram: {sankey_feature} Flow", template=plotly_template)
            st.plotly_chart(fig_sankey, use_container_width=True)
    with tab4:
        st.subheader("🖱️ Interactive Plots")
        st.markdown("<div class='tooltip'>What is this?<span class='tooltiptext'>Colorful charts like sunburst to explore soil hierarchies. Example: Loamy with 40.7% clay, pH 5.66 is acidic.</span></div>", unsafe_allow_html=True)
        line_feature = st.selectbox("Select Feature for Line Plot", all_features)
        fig_line = px.line(
            filtered_df.sort_values(by=line_feature),
            x=line_feature, y=selected_features[1] if len(selected_features) > 1 else 'Organic_Carbon',
            color='Soil_Type', title=f"Line Plot: {line_feature}",
            color_discrete_sequence=px.colors.qualitative.Prism,
            template=plotly_template
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        area_y_feature = st.selectbox("Select Feature for Area Plot", selected_features or all_features)
        fig_area = px.area(
            filtered_df, x='Soil_pH', y=area_y_feature, color='Soil_Type',
            title=f"Area Plot: Soil pH vs {area_y_feature}",
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            template=plotly_template
        )
        st.plotly_chart(fig_area, use_container_width=True)
        
        sunburst_value = st.selectbox("Select Value for Sunburst Plot", ['Sand_Content', 'Silt_Content', 'Organic_Carbon'])
        fig_sunburst = px.sunburst(
            filtered_df, path=['Soil_Type', 'Clay_Content'], values=sunburst_value,
            title="Sunburst Plot: Soil Type Hierarchy", color='Soil_pH',
            color_continuous_scale='Rainbow', template=plotly_template
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
        if kaleido:
            buf = BytesIO()
            try:
                fig_sunburst.write_image(buf, format="png")
                st.download_button("Download Sunburst Plot", buf.getvalue(), "sunburst_plot.png", "image/png")
            except Exception:
                st.warning("Cannot export sunburst plot.")
        
        treemap_value = st.selectbox("Select Value for Treemap", ['Sand_Content', 'Silt_Content', 'Organic_Carbon'])
        fig_treemap = px.treemap(
            filtered_df, path=['Soil_Type', 'Clay_Content'], values=treemap_value,
            title="Treemap: Soil Type Hierarchy", color='Soil_pH',
            color_continuous_scale='Plasma', template=plotly_template
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    with tab5:
        st.subheader("⚙️ Predict Soil Type")
        st.markdown("<div class='tooltip'>What is this?<span class='tooltiptext'>Enter soil values to predict if it’s Sandy, Loamy, or Clayey, with crop suggestions!</span></div>", unsafe_allow_html=True)
        with st.sidebar:
            st.header("Input Features")
            soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.0)
            organic_carbon = st.slider("Organic Carbon (%)", 0.0, 10.0, 1.0)
            clay_content = st.slider("Clay Content (%)", 0.0, 100.0, 30.0)
            sand_content = st.slider("Sand Content (%)", 0.0, 100.0, 40.0)
            silt_content = st.slider("Silt Content (%)", 0.0, 100.0, 30.0)
            ec = st.slider("EC (dS/m)", 0.0, 5.0, 1.0)
        input_df = pd.DataFrame({
            'Soil_pH': [soil_ph], 'Organic_Carbon': [organic_carbon], 'Clay_Content': [clay_content],
            'Sand_Content': [sand_content], 'Silt_Content': [silt_content], 'EC': [ec]
        })
        input_df['Clay_Sand_Ratio'] = input_df['Clay_Content'] / (input_df['Sand_Content'] + 1e-6)
        input_df['Texture_Sum'] = input_df['Clay_Content'] + input_df['Sand_Content'] + input_df['Silt_Content']
        input_df['pH_EC_Interaction'] = input_df['Soil_pH'] * input_df['EC']
        input_df['Organic_Texture_Ratio'] = input_df['Organic_Carbon'] / (input_df['Texture_Sum'] + 1e-6)
        X_input = input_df[feature_columns]
        try:
            X_input_scaled = scaler.transform(X_input)
            prediction_encoded = model.predict(X_input_scaled)[0]
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]
            st.metric(label="Predicted Soil Type", value=prediction, delta="🌿 Predicted")
            crops, advice, status = recommend_crops(prediction, soil_ph)
            st.write(f"**Crop Recommendations**: {', '.join(crops)}")
            st.write(f"**Advice**: {advice}")
            st.write(f"**pH Status**: {status}")
            probs = model.predict_proba(X_input_scaled)[0]
            prob_df = pd.DataFrame({'Soil Type': label_encoder.classes_, 'Probability': probs})
            fig_prob = px.bar(
                prob_df, x='Soil Type', y='Probability', title="Prediction Probabilities",
                color='Soil Type', color_discrete_sequence=px.colors.qualitative.Vivid,
                template=plotly_template
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append({
                'Soil_pH': soil_ph, 'Organic_Carbon': organic_carbon, 'Clay_Content': clay_content,
                'Sand_Content': sand_content, 'Silt_Content': silt_content, 'EC': ec,
                'Predicted_Soil_Type': prediction
            })
            st.write("Prediction History:")
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)
            history_csv = pd.DataFrame(st.session_state.prediction_history).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction History 📥",
                data=history_csv,
                file_name="prediction_history.csv",
                mime="text/csv"
            )
            st.header("Batch Prediction")
            batch_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                batch_df['Clay_Sand_Ratio'] = batch_df['Clay_Content'] / (batch_df['Sand_Content'] + 1e-6)
                batch_df['Texture_Sum'] = batch_df['Clay_Content'] + batch_df['Sand_Content'] + batch_df['Silt_Content']
                batch_df['pH_EC_Interaction'] = batch_df['Soil_pH'] * batch_df['EC']
                batch_df['Organic_Texture_Ratio'] = batch_df['Organic_Carbon'] / (batch_df['Texture_Sum'] + 1e-6)
                X_batch = batch_df[feature_columns]
                X_batch_scaled = scaler.transform(X_batch)
                batch_predictions = label_encoder.inverse_transform(model.predict(X_batch_scaled))
                batch_df['Predicted_Soil_Type'] = batch_predictions
                st.write("Batch Prediction Results:")
                st.dataframe(batch_df, use_container_width=True)
                batch_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Batch Predictions 📥",
                    data=batch_csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
        except ValueError as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Check if input features match the model's requirements.")

    with tab6:
        st.subheader("📈 Model Explanation (SHAP)")
        if st.button("Generate SHAP Explanation"):
            st.write("Generating SHAP values...")
            shap_df = filtered_df.copy()
            shap_df['Clay_Sand_Ratio'] = shap_df['Clay_Content'] / (shap_df['Sand_Content'] + 1e-6)
            shap_df['Texture_Sum'] = shap_df['Clay_Content'] + shap_df['Sand_Content'] + shap_df['Silt_Content']
            shap_df['pH_EC_Interaction'] = shap_df['Soil_pH'] * shap_df['EC']
            shap_df['Organic_Texture_Ratio'] = shap_df['Organic_Carbon'] / (shap_df['Texture_Sum'] + 1e-6)
            X_scaled = scaler.transform(shap_df[feature_columns])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            st.write("SHAP Summary Plot")
            fig_shap_summary = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, shap_df[feature_columns], show=False)
            st.pyplot(fig_shap_summary)
            st.write("SHAP Force Plot")
            try:
                # Predict class for the first sample
                first_sample = X_scaled[0:1]
                predicted_class = model.predict(first_sample)[0]
                # Verify feature alignment
                if len(shap_values[predicted_class][0]) != len(feature_columns):
                    st.error(f"Dimension mismatch: SHAP values have {len(shap_values[predicted_class][0])} features, but feature_columns has {len(feature_columns)} features: {feature_columns}")
                    st.write("Forcing model retraining with correct features...")
                    # Delete old model to force retraining
                    model_path = 'improved_models/model_random_forest_(smote).pkl'
                    scaler_path = 'improved_models/scaler.pkl'
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    if os.path.exists(scaler_path):
                        os.remove(scaler_path)
                    model, scaler, label_encoder = get_model(n_estimators)
                    X_scaled = scaler.transform(shap_df[feature_columns])
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_scaled)
                # Use SHAP values and expected value for the predicted class
                shap_values_for_class = shap_values[predicted_class]
                expected_value = explainer.expected_value[predicted_class]
                fig_shap_force = plt.figure(figsize=(10, 4))
                shap.initjs()
                shap.force_plot(
                    expected_value,
                    shap_values_for_class[0],
                    shap_df[feature_columns].iloc[0],
                    matplotlib=True,
                    feature_names=feature_columns
                )
                st.pyplot(fig_shap_force)
                st.write(f"Force plot shows explanation for predicted class: {label_encoder.inverse_transform([predicted_class])[0]}")
            except Exception as e:
                st.error(f"Error generating SHAP force plot: {str(e)}")
                st.write(f"Debug info: SHAP values shape: {shap_values[predicted_class][0].shape}, Features: {len(feature_columns)} ({feature_columns})")
                st.write("Try deleting 'improved_models/' folder and rerunning to retrain the model.")
    with tab7:
        st.subheader("🌟 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        fig_importance = px.bar(
            importance_df, x='Importance', y='Feature', orientation='h',
            title="Feature Importance", color='Importance',
            color_continuous_scale='Viridis', template=plotly_template
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)

    with tab8:
        st.subheader("📊 Model Performance")
        st.markdown("<div class='tooltip'>What is this?<span class='tooltiptext'>Shows model accuracy, confusion matrix, classification report, ROC curves, and precision-recall curves.</span></div>", unsafe_allow_html=True)
        if filtered_df.empty:
            st.warning("No data available after applying filters. Please adjust sidebar filters (e.g., select more soil types or widen pH range).")
        else:
            data = filtered_df.copy()
            data['Clay_Sand_Ratio'] = data['Clay_Content'] / (data['Sand_Content'] + 1e-6)
            data['Texture_Sum'] = data['Clay_Content'] + data['Sand_Content'] + data['Silt_Content']
            data['pH_EC_Interaction'] = data['Soil_pH'] * data['EC']
            data['Organic_Texture_Ratio'] = data['Organic_Carbon'] / (data['Texture_Sum'] + 1e-6)
            y = label_encoder.transform(data['Soil_Type'])
            X = data[feature_columns]
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
            # Accuracy
            accuracy = accuracy_score(y, y_pred)
            st.metric("Model Accuracy", f"{accuracy:.2%}", delta="🌟 Overall Performance")
            # Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            fig_cm = px.imshow(
                cm, x=label_encoder.classes_, y=label_encoder.classes_,
                text_auto=True, title="Confusion Matrix",
                color_continuous_scale='Blues', template=plotly_template
            )
            fig_cm.update_layout(
                xaxis_title="Predicted Soil Type",
                yaxis_title="True Soil Type",
                font=dict(size=14)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            # Classification Report
            st.write("Classification Report")
            clf_report = classification_report(y, y_pred, target_names=label_encoder.classes_, output_dict=True)
            clf_df = pd.DataFrame(clf_report).transpose().round(3)
            st.dataframe(clf_df, use_container_width=True)
            csv = clf_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Classification Report 📥",
                data=csv,
                file_name="classification_report.csv",
                mime="text/csv",
                help="Download precision, recall, and F1-score details."
            )
            # ROC Curves and AUC
            st.write("ROC Curves and AUC Scores")
            y_proba = model.predict_proba(X_scaled)
            fig_roc = go.Figure()
            for i, class_name in enumerate(label_encoder.classes_):
                fpr, tpr, _ = roc_curve(y == i, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines', name=f'{class_name} (AUC = {roc_auc:.3f})',
                    line=dict(width=2, color=px.colors.qualitative.Vivid[i])
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')
            ))
            fig_roc.update_layout(
                title="ROC Curves by Soil Type",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template=plotly_template,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                font=dict(size=14)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            # Precision-Recall Curves
            st.write("Precision-Recall Curves")
            fig_pr = go.Figure()
            for i, class_name in enumerate(label_encoder.classes_):
                precision, recall, _ = precision_recall_curve(y == i, y_proba[:, i])
                pr_auc = auc(recall, precision)
                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision, mode='lines', name=f'{class_name} (AUC = {pr_auc:.3f})',
                    line=dict(width=2, color=px.colors.qualitative.Vivid[i])
                ))
            fig_pr.update_layout(
                title="Precision-Recall Curves by Soil Type",
                xaxis_title="Recall",
                yaxis_title="Precision",
                template=plotly_template,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                font=dict(size=14)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    # Expanded Tutorial with more details
    with st.expander("📚 Expanded Tutorial for Beginners", expanded=True):
        st.markdown("""
        **Welcome to the #1 Vibrant Soil Dashboard!** This tool is designed for beginners like you. Here's how to use it:
        - **Data Overview**: View soil data. Hover over columns for details.
        - **Basic EDA**: Simple charts (histograms, scatter). Choose a theme in the sidebar to see changes.
        - **Advanced EDA**: 16 stunning charts (e.g., sunburst for soil hierarchies, word cloud for soil types). Select 2+ features to see plots.
        - **Interactive Plots**: Fun charts like sunburst. Example: Loamy with 40.7% clay, pH 5.66 is slightly acidic (red color).
        - **Prediction**: Enter values to predict soil type. Get crop suggestions (e.g., teff for Loamy, adjust pH if acidic).
        - **Model Explanation**: SHAP plots to understand why the model predicts a soil type.
        - **Feature Importance**: See which features (e.g., clay content) matter most.
        - **Model Performance**: Check accuracy (how often the model is right).
        
        **Tips for Amhara Farmers**: Use pH filter to find soils for teff (pH 6.0–7.0). Upload your soil data CSV for personalized predictions!
        **Theme Changes**: If themes don't change, clear your browser cache (Ctrl+Shift+Delete in Chrome) or try incognito mode.
        """)

    # Footer
    st.markdown("---")
    st.markdown("Developed for vibrant soil analysis in Amhara, Ethiopia. 🌾 Feedback welcome!")
    with st.expander("💬 Provide Feedback"):
        feedback = st.text_area("Your thoughts on the dashboard")
        if st.button("Submit Feedback"):
            with open("feedback.txt", "a") as f:
                f.write(f"{feedback}\n")
            st.success("Thank you for your feedback! 🙌")

else:
    st.error("Unable to load data or train model. Check file paths and dependencies.")
