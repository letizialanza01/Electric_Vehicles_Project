##########import libraries 
import streamlit as st 
import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
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
        
    st.markdown('''
                In this case, since the 'County', 'City' and 'Electric Utility' columns contain categorical variables and not numeric variables, it would not be correct to replace missing values with mathematical operations such as mode or median. So the null values have been dropped.''')


##########exlporatory data analysis 
elif current_tab == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis') 
    
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(['Top companies', 'TESLA', 'CHEVROLET', 'NISSAN', 'FORD', 'KIA', 'BMW', 'TOYOTA', 'VOLKSWAGEN', 'JEEP', 'HYUNDAI'])

with tab1:
    vehicles_counts_by_name = electric_vehicles.groupby('Make').size().sort_values(ascending = False)  #group by 'Make', count occurrences, and sort by the count in descending order

    #select the top 10 companies
    top_companies = vehicles_counts_by_name.index[:10]
    top_values = vehicles_counts_by_name.values[:10]

    #print the top 10 companies and their values
    for company, count in zip(top_companies, top_values):
        print(f'For {company}, the number of electric vehicles registered are {count}')

    #plot the 10 companies
    plt.figure(figsize = (12, 6))
    sns.barplot(x = top_companies, y = top_values, edgecolor = 'black', linewidth = 1, alpha = 0.7, palette = 'Accent', hue = list(top_companies)[:10], dodge = False, legend = False)
    plt.xlabel('Companies', fontsize = 12, fontweight = 'bold', color = 'blue')
    plt.ylabel('Values', fontsize = 12, fontweight = 'bold', color = 'blue')
    plt.title('Top electric car companies', fontsize = 18, fontweight = 'bold', color = 'red')
    plt.xticks(rotation = 45, fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)

    plt.tight_layout()
    plt.show()
    
    #define a colormap to generate distinctive colors
    num_companies = len(top_companies)
    cmap = cm.get_cmap('tab20', num_companies * 2) 
    num_cols = 2  #two columns and rows for the plots
    num_rows = (num_companies + 1) // num_cols  #calculate the necessary number of rows

    plt.figure(figsize = (12, 6 * num_rows))

    for idx, company in enumerate(top_companies):
        company_data = electric_vehicles[electric_vehicles['Make'] == company]
        total_sales = company_data.shape[0]
        model_sales_counts = company_data.groupby('Model').size().sort_values(ascending = False)
        top_selling_model = model_sales_counts.index[0]
        top_selling_model_count = model_sales_counts.iloc[0]

        sales_data = [top_selling_model_count, total_sales - top_selling_model_count]  #prepare the data for the pie chart
        labels = [top_selling_model, 'Other models']
        colors = [cmap(idx * 2), cmap(idx * 2 + 1)]  #select a pair of distinctive colors for each pie chart 

        #create the pie chart for total and top model sales 
        plt.subplot(num_rows, num_cols, idx + 1) 
        plt.pie(sales_data, labels = labels, autopct = '%1.1f%%', colors = colors, textprops = {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'})
        plt.title(f'Total vs Top Model Sales for {company}', fontsize = 18, fontweight = 'bold', color = 'red')

    plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
    plt.tight_layout()
    plt.show()