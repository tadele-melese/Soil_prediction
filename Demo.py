# import streamlit as st
# import pandas as pd

# # Inject CSS to customize tab colors
# st.markdown(
#     """
#     <style>
#     /* Change background and text color of tabs */
#     .stTabs [role="tablist"] button[role="tab"] {
#         background-color: #f0f2f6;  /* default gray */
#         color: black;
#         border-radius: 10px 10px 0px 0px;
#         padding: 8px;
#     }
    
#     /* Active tab */
#     .stTabs [role="tablist"] button[aria-selected="true"] {
#         background-color: #4CAF50; /* green for active */
#         color: white;
#     }

#     /* First tab */
#     .stTabs [role="tablist"] button:nth-child(1) {
#         background-color: #2196F3; /* blue */
#         color: white;
#     }

#     /* Second tab */
#     .stTabs [role="tablist"] button:nth-child(2) {
#         background-color: #FF9800; /* orange */
#         color: white;
#     }

#     /* Third tab */
#     .stTabs [role="tablist"] button:nth-child(3) {
#         background-color: #9C27B0; /* purple */
#         color: white;
#     }

#       /* Forth tab */
#     .stTabs [role="tablist"] button:nth-child(4) {
#         background-color: #9C27B0; /* purple */
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# st.title('Dashboard Demo')
# st.markdown("""
#             ### Developed by TM
            # #*Year 2025*
#             """)

# tab1, tab2,tab3, tab4 = st.tabs(['Load Data', 'Multi_colomn Display', 'Filter Dta', 'Visuaize'])

# with tab1:
#     data = pd.read_csv("simulated_soil_data.csv")
#     data
#     st.line_chart(data=data['Soil_pH'], color='#ffaa00')
#     # st.bar_chart(data=data)
#     st.dataframe(data=data)
# with tab2:
#     columns = st.multiselect(label='Select colomns to display', 
#                             default= data.columns.tolist(),
#                             options=data.columns.tolist())
#     st.dataframe(data[columns])



# st.sidebar.title("Filter Bar")
# # Filter data using sllidebare
# pH_min, pH_max = st.sidebar.slider("Soil_pH range", float(data['Soil_pH'].min()), float(data['Soil_pH'].max()), (0.0,9.0))

# EC_min, EC_max = st.sidebar.slider("EC_range", float(data['EC'].min()), float(data['EC'].max()), (0.0, 40.0))

# with tab3:
#     # Filtering data 
#     filtered_data = data[
#         (data['Soil_pH'] >= pH_min) & (data['Soil_pH'] <= pH_max) &
#         (data['EC'] >= EC_min) & (data['EC'] <= EC_max)
#     ]

#     st.write("Filtered Data")
#     st.dataframe(filtered_data)

# with tab4:
#     # Visualize
#     st.line_chart(filtered_data[['Soil_pH', 'EC']])

