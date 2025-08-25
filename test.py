import streamlit as st
import pandas as pd

import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-color: #c3c3c3;  /* light gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='color:red'>Hello</h1>", unsafe_allow_html=True)
# Title
st.title("Demonstration Dashboard")
st.write("This dashboards visuallize soil parameters, analysis and prediction")
tab1,tab2,tab3,tab4 = st.tabs(['Basic Statistcs', 'Visualization', 'Model Training', 'Performance'])
# Load the dataset
data =pd.read_csv("simulated_soil_data.csv")
data
with tab1:
    st.title('Basic Statisics Information')
    st.markdown("""
                # 'Description',
                *Inferential*
                """)
with tab2:
    st.title('Visualzation section')
    st.dataframe(data)
    st.scatter_chart(data, x='EC',y='Soil_pH')

with tab3:
    st.bar_chart(data[['EC','Soil_pH', 'Organic_Carbon', 'Clay_Content']])
with tab4:
    st.write('Model performance will be displayed here!  Stay tuned!!!!!!!!!!!!!!!!')
    st.bar_chart(data, x="category", y="sales")

st.dataframe(data)

st.line_chart(data['EC'])
st.bar_chart(data[['EC','Soil_pH', 'Organic_Carbon', 'Clay_Content']])
st.scatter_chart(data, x='EC',y='Soil_pH')
# to show many colomns from the table
# Create side bar

st.sidebar.title("Data Filter Bar")
ph_min, ph_max = st.sidebar.slider('Soil pH range', float(data['Soil_pH'].min()), float(data['Soil_pH'].max()), (1.0, 9.0))

