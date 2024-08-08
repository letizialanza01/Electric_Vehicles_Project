##########import libraries 
import streamlit as st 
import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import io 
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

##########import dataset 
electric_vehicles = pd.read_csv('Electric_Vehicle_Population_data.csv')

##########
#set tabs for the chapters 
##########

tab_names = ['Introduction', 'Cleaning', 'Exploratory Data Analysis', 'Correlation', 'Modelling with ML Algorithms']
current_tab = st.sidebar.selectbox('Summary', tab_names)
st.sidebar.markdown(
    """
    **Letizia Lanza** \n
    My page on GitHub: [GitHub](https://github.com/letizialanza01)   
    My LinkedIn profile: [LinkedIn](linkedin.com/in/letizia-lanza)
    """
)


##########introduction
if current_tab == 'Introduction':
    st.markdown('<h1 style = "text-align: center;"> Electric cars currently registered</h1>', unsafe_allow_html = True)
    st.subheader('Programming for Data Science: Final Project')
    st.markdown('''
                **Author:** Letizia Lanza
                ''')
    st.markdown('''
                The dataset is about the Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) that are currently registered through Washington State Department of Licensing (DOL) and on the roads of The United States of America. (Updated: 22/06/2024) \n 
                **Data Source:** https://www.kaggle.com/datasets/adarshde/electric-vehicle-population-dataset/data
                ''')

    
    selected_columns = st.multiselect('Explore the Electric cars dataset by selecting columns', electric_vehicles.columns)
    if selected_columns:
        columns_df = electric_vehicles.loc[:, selected_columns]
        st.dataframe(columns_df.head(15))
    else: 
        st.dataframe(electric_vehicles.head(15))
        
        
        
    st.write('General information about the DataFrame')
    #creating a buffer to capture information on the electric vehicles dataframe
    buffer = io.StringIO()
    electric_vehicles.info(buf = buffer)
    s = buffer.getvalue()
    #show multiselect to select columns to display 
    selected_columns1 = st.multiselect('Select the variables', electric_vehicles.columns.tolist(), default = electric_vehicles.columns.tolist())
    
    #if columns are selected, it shows information only for those columns 
    if selected_columns1:
        selected_info_buffer = io.StringIO()
        electric_vehicles[selected_columns1].info(buf = selected_info_buffer)
        selected_info = selected_info_buffer.getvalue()
        st.text(selected_info)
    else: 
        #otherwise, it shows information for all columns
        st.text(s)
        
        
        
##########cleaning        
elif current_tab == 'Cleaning':
    st.title('Cleaning the NA values')
    
    st.write('Before proceeding with the analysis, the null values in the dataset were analyzed and then eliminated.')
    
    tab1, tab2 = st.tabs(['NA values', 'Cleaning'])
    
    with tab1:
        st.write('Scroll down to find null values')
        #calculates the count of missing values 
        electric_vehicles = electric_vehicles.drop(['VIN (1-10)', 'Postal Code', 'Base MSRP', 'Legislative District', 'DOL Vehicle ID', 'Vehicle Location', '2020 Census Tract'], axis = 1)
        missing_values_count = electric_vehicles.isna().sum()
        
        #create a new DataFrame with the count and percentage of missing values 
        missing_df = pd.DataFrame({
            'Variable': missing_values_count.index,
            'NA Values': missing_values_count.values
        })
        
        #show the DataFrame of missing values 
        st.write(missing_df)
        
    with tab2:
        st.markdown('''
                    In this case, since the 'County', 'City' and 'Electric Utility' columns contain categorical variables and not numeric variables, it would not be correct to replace missing values with mathematical operations such as mode or median. So the null values have been dropped.''')


##########exlporatory data analysis 
elif current_tab == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis') 