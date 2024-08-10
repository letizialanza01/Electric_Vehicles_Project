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
    st.markdown('<h1 style = "text-align: center;"> Registered electric cars in the USA </h1>', unsafe_allow_html = True)
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
    
    st.write('The chosen dataset has multiple data, therefore, in order to make the exploratory data analysis more understandable and meaningful, it was chosen to focus only on the data for the 10 companies that attest to higher values of registered cars. Furthermore it is assumed that the most registered car models are at the same time the units sold/the best selling models.')
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(['Top companies', 'TESLA', 'CHEVROLET', 'NISSAN', 'FORD', 'KIA', 'BMW', 'TOYOTA', 'VOLKSWAGEN', 'JEEP', 'HYUNDAI'])

    with tab1:
        vehicles_counts_by_name = electric_vehicles.groupby('Make').size().sort_values(ascending = False)  #group by 'Make', count occurrences, and sort by the count in descending order

        #select the top 10 companies
        top_companies = vehicles_counts_by_name.index[:10]
        top_values = vehicles_counts_by_name.values[:10]

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
        st.pyplot(plt.gcf())
        
        #print the top 10 companies and their values
        for company, count in zip(top_companies, top_values):
            st.write(f'For {company}, the number of electric vehicles registered are {count}')
        
        st.write('Below is intended to highlight the percentage of best-selling model sales for each of the top 10 car companies in comparison to total sales.')
        
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
        st.pyplot(plt.gcf())
        
        st.write('Still, shown below are the 15 cities with the highest number of electric cars registered.')
        
        filtered_data = electric_vehicles[electric_vehicles['Model Year'] >= 2005]  #filter the dataframe to include only rows where 'Model Year' is 2005 or later
        city_counts = filtered_data['City'].value_counts().reset_index()  #count occurrences of each city in the filtered data and reset index
        city_counts.columns = ['City', 'Count']
        city_counts = city_counts.sort_values(by = 'Count', ascending = False)
        top_15_cities = city_counts.head(15)
        palette = sns.color_palette('muted', len(top_15_cities))

        #create a vertical bar chart
        plt.figure(figsize = (12, 6))
        bars = plt.bar(top_15_cities['City'], top_15_cities['Count'], color = palette)
        plt.title('Top 15 cities by number of electric vehicles sold from 2005', fontsize = 18, fontweight = 'bold', color = 'red')
        plt.xlabel('City', fontsize = 12, fontweight = 'bold', color = 'blue')
        plt.ylabel('Number of cars sold', fontsize = 12, fontweight = 'bold', color = 'blue')
        plt.xticks(rotation = 45, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        st.write('Seattle is definitely the city with more registered electric cars than any other city. This is the result of a combination of favorable environmental policies, economic incentives, community environmental awareness, infrastructure support and a local sustainability-oriented culture.')
        
    with tab2:
        tesla_data = electric_vehicles[electric_vehicles['Make'] == 'TESLA']  #filter the data for Tesla models
        tesla_top_model_data = tesla_data.groupby('Model').size().sort_values(ascending = False)  #find the top Tesla model
        top_tesla_model = tesla_top_model_data.index[0]
        top_tesla_model_data = tesla_data[tesla_data['Model'] == top_tesla_model]  #filter the data for the top model
        tesla_yearly_sales = top_tesla_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model

        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(tesla_yearly_sales.index, tesla_yearly_sales.values, color = '#FFA500', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_tesla_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(tesla_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        tesla_data = electric_vehicles[electric_vehicles['Make'] == 'TESLA']  #filter the data for Tesla models
        tesla_top_model_data = tesla_data.groupby('Model').size().sort_values(ascending = False)  #find the top Tesla model
        top_tesla_model = tesla_top_model_data.index[0]
        top_tesla_model_data = tesla_data[tesla_data['Model'] == top_tesla_model]  #filter the data for the top model
        tesla_yearly_sales = top_tesla_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        tesla_sales_df = pd.DataFrame(tesla_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        tesla_sales_df.columns = ['Model Year', 'Units Sold']
        tesla_sales_df['Model Year'] = tesla_sales_df['Model Year'].astype(str)
        tesla_sales_df['Units Sold'] = tesla_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(tesla_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        tesla_years_of_interest = [2020, 2021, 2022, 2023, 2024]  #define the years of interest
        tesla_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for year_tesla in tesla_years_of_interest:
            tesla_yearly_data = top_tesla_model_data[top_tesla_model_data['Model Year'] == year_tesla]  #filter data for the specific year
            tesla_city_sales_data = tesla_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            tesla_top_city = tesla_city_sales_data.loc[tesla_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            tesla_city_sales['Year'].append(year_tesla)
            tesla_city_sales['City'].append(tesla_top_city['City'])
            tesla_city_sales['Number of Sales'].append(tesla_top_city['Number of Sales'])

        tesla_city_sales_df = pd.DataFrame(tesla_city_sales)  #create a DataFrame from the results
        st.table(tesla_city_sales_df)
        
    with tab3:
        chevrolet_data = electric_vehicles[electric_vehicles['Make'] == 'CHEVROLET']  #filter the data for Chevrolet models
        chevrolet_top_model_data = chevrolet_data.groupby('Model').size().sort_values(ascending = False)  #find the top Chevrolet model
        top_chevrolet_model = chevrolet_top_model_data.index[0]
        top_chevrolet_model_data = chevrolet_data[chevrolet_data['Model'] == top_chevrolet_model]  #filter the data for the top model
        chevrolet_yearly_sales = top_chevrolet_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(chevrolet_yearly_sales.index, chevrolet_yearly_sales.values, color = '#800080', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_chevrolet_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(chevrolet_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        chevrolet_data = electric_vehicles[electric_vehicles['Make'] == 'CHEVROLET']  #filter the data for Chevrolet models
        chevrolet_top_model_data = chevrolet_data.groupby('Model').size().sort_values(ascending = False)  #find the top Chevrolet model
        top_chevrolet_model = chevrolet_top_model_data.index[0]
        top_chevrolet_model_data = chevrolet_data[chevrolet_data['Model'] == top_chevrolet_model]  #filter the data for the top model
        chevrolet_yearly_sales = top_chevrolet_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        chevrolet_sales_df = pd.DataFrame(chevrolet_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        chevrolet_sales_df.columns = ['Model Year', 'Units Sold']
        chevrolet_sales_df['Model Year'] = chevrolet_sales_df['Model Year'].astype(str)
        chevrolet_sales_df['Units Sold'] = chevrolet_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(chevrolet_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        chevrolet_years_of_interest = [2017, 2018, 2019, 2020, 2021, 2022, 2023]  #define the years of interest
        chevrolet_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for chevrolet_year in chevrolet_years_of_interest:
            chevrolet_yearly_data = top_chevrolet_model_data[top_chevrolet_model_data['Model Year'] == chevrolet_year]  #filter data for the specific year
            chevrolet_city_sales_data = chevrolet_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            chevrolet_top_city = chevrolet_city_sales_data.loc[chevrolet_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            chevrolet_city_sales['Year'].append(chevrolet_year)
            chevrolet_city_sales['City'].append(chevrolet_top_city['City'])
            chevrolet_city_sales['Number of Sales'].append(chevrolet_top_city['Number of Sales'])

        chevrolet_city_sales_df = pd.DataFrame(chevrolet_city_sales)  #create a DataFrame from the results
        st.table(chevrolet_city_sales_df)
        
    with tab4:
        nissan_data = electric_vehicles[electric_vehicles['Make'] == 'NISSAN']  #filter the data for Nissan models
        nissan_top_model_data = nissan_data.groupby('Model').size().sort_values(ascending = False)  #find the top Nissan model
        top_nissan_model = nissan_top_model_data.index[0]
        top_nissan_model_data = nissan_data[nissan_data['Model'] == top_nissan_model]  #filter the data for the top model
        nissan_yearly_sales = top_nissan_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(nissan_yearly_sales.index, nissan_yearly_sales.values, color = '#A52A2A', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_nissan_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(nissan_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        nissan_data = electric_vehicles[electric_vehicles['Make'] == 'NISSAN']  #filter the data for Nissan models
        nissan_top_model_data = nissan_data.groupby('Model').size().sort_values(ascending = False)  #find the top Nissan model
        top_nissan_model = nissan_top_model_data.index[0]
        top_nissan_model_data = nissan_data[nissan_data['Model'] == top_nissan_model]  #filter the data for the top model
        nissan_yearly_sales = top_nissan_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        nissan_sales_df = pd.DataFrame(nissan_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        nissan_sales_df.columns = ['Model Year', 'Units Sold']
        nissan_sales_df['Model Year'] = nissan_sales_df['Model Year'].astype(str)
        nissan_sales_df['Units Sold'] = nissan_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(nissan_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        nissan_years_of_interest = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  #define the years of interest
        nissan_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for nissan_year in nissan_years_of_interest:
            nissan_yearly_data = top_nissan_model_data[top_nissan_model_data['Model Year'] == nissan_year]  #filter data for the specific year
            nissan_city_sales_data = nissan_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            nissan_top_city = nissan_city_sales_data.loc[nissan_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            nissan_city_sales['Year'].append(nissan_year)
            nissan_city_sales['City'].append(nissan_top_city['City'])
            nissan_city_sales['Number of Sales'].append(nissan_top_city['Number of Sales'])

        nissan_city_sales_df = pd.DataFrame(nissan_city_sales)  #create a dataframe from the results
        st.table(nissan_city_sales_df)
        
    with tab5:
        ford_data = electric_vehicles[electric_vehicles['Make'] == 'FORD']  #filter the data for Ford models
        ford_top_model_data = ford_data.groupby('Model').size().sort_values(ascending = False)  #find the top Ford model
        top_ford_model = ford_top_model_data.index[0]
        top_ford_model_data = ford_data[ford_data['Model'] == top_ford_model]  #filter the data for the top model
        ford_yearly_sales = top_ford_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
       
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(ford_yearly_sales.index, ford_yearly_sales.values, color = '#FF00FF', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_ford_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(ford_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        ford_data = electric_vehicles[electric_vehicles['Make'] == 'FORD']  #filter the data for Ford models
        ford_top_model_data = ford_data.groupby('Model').size().sort_values(ascending = False)  #find the top Ford model
        top_ford_model = ford_top_model_data.index[0]
        top_ford_model_data = ford_data[ford_data['Model'] == top_ford_model]  #filter the data for the top model
        ford_yearly_sales = top_ford_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        ford_sales_df = pd.DataFrame(ford_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        ford_sales_df.columns = ['Model Year', 'Units Sold']
        ford_sales_df['Model Year'] = ford_sales_df['Model Year'].astype(str)
        ford_sales_df['Units Sold'] = ford_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(ford_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        ford_years_of_interest = [2021, 2022, 2023, 2024]  #define the years of interest
        ford_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for ford_year in ford_years_of_interest:
            ford_yearly_data = top_ford_model_data[top_ford_model_data['Model Year'] == ford_year]  #filter data for the specific year
            ford_city_sales_data = ford_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            ford_top_city = ford_city_sales_data.loc[ford_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            ford_city_sales['Year'].append(ford_year)
            ford_city_sales['City'].append(ford_top_city['City'])
            ford_city_sales['Number of Sales'].append(ford_top_city['Number of Sales'])

        ford_city_sales_df = pd.DataFrame(ford_city_sales)  #create a DataFrame from the results
        st.table(ford_city_sales_df)
        
    with tab6:
        kia_data = electric_vehicles[electric_vehicles['Make'] == 'KIA']  #filter the data for Kia models
        kia_top_model_data = kia_data.groupby('Model').size().sort_values(ascending = False)  #find the top Kia model
        top_kia_model = kia_top_model_data.index[0]
        top_kia_model_data = kia_data[kia_data['Model'] == top_kia_model]  #filter the data for the top model
        kia_yearly_sales = top_kia_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
    
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(kia_yearly_sales.index, kia_yearly_sales.values, color = '#F4A460', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_kia_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(kia_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        kia_data = electric_vehicles[electric_vehicles['Make'] == 'KIA']  #filter the data for Kia models
        kia_top_model_data = kia_data.groupby('Model').size().sort_values(ascending = False)  #find the top Kia model
        top_kia_model = kia_top_model_data.index[0]
        top_kia_model_data = kia_data[kia_data['Model'] == top_kia_model]  #filter the data for the top model
        kia_yearly_sales = top_kia_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        kia_sales_df = pd.DataFrame(kia_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        kia_sales_df.columns = ['Model Year', 'Units Sold']
        kia_sales_df['Model Year'] = kia_sales_df['Model Year'].astype(str)
        kia_sales_df['Units Sold'] = kia_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(kia_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        kia_years_of_interest = [2018, 2019, 2020, 2021, 2022, 2023, 2024]  #define the years of interest
        kia_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for kia_year in kia_years_of_interest:
            kia_yearly_data = top_kia_model_data[top_kia_model_data['Model Year'] == kia_year]  #filter data for the specific year
            kia_city_sales_data = kia_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            kia_top_city = kia_city_sales_data.loc[kia_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            kia_city_sales['Year'].append(kia_year)
            kia_city_sales['City'].append(kia_top_city['City'])
            kia_city_sales['Number of Sales'].append(kia_top_city['Number of Sales'])

        kia_city_sales_df = pd.DataFrame(kia_city_sales)  #create a DataFrame from the results
        st.table(kia_city_sales_df)
        
    with tab7: 
        bmw_data = electric_vehicles[electric_vehicles['Make'] == 'BMW']  #filter the data for BMW models
        bmw_top_model_data = bmw_data.groupby('Model').size().sort_values(ascending = False)  #find the top BMW model
        top_bmw_model = bmw_top_model_data.index[0]
        top_bmw_model_data = bmw_data[bmw_data['Model'] == top_bmw_model]  #filter the data for the top model
        bmw_yearly_sales = top_bmw_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(bmw_yearly_sales.index, bmw_yearly_sales.values, color = 'green', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_bmw_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(bmw_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        bmw_data = electric_vehicles[electric_vehicles['Make'] == 'BMW']  #filter the data for BMW models
        bmw_top_model_data = bmw_data.groupby('Model').size().sort_values(ascending = False)  #find the top BMW model
        top_bmw_model = bmw_top_model_data.index[0]
        top_bmw_model_data = bmw_data[bmw_data['Model'] == top_bmw_model]  #filter the data for the top model
        bmw_yearly_sales = top_bmw_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        bmw_sales_df = pd.DataFrame(bmw_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        bmw_sales_df.columns = ['Model Year', 'Units Sold']
        bmw_sales_df['Model Year'] = bmw_sales_df['Model Year'].astype(str)
        bmw_sales_df['Units Sold'] = bmw_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(bmw_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        bmw_years_of_interest = [2016, 2017, 2018, 2021, 2022, 2023, 2024]  #define the years of interest
        bmw_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for bmw_year in bmw_years_of_interest:
            bmw_yearly_data = top_bmw_model_data[top_bmw_model_data['Model Year'] == bmw_year]  #filter data for the specific year
            bmw_city_sales_data = bmw_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            bmw_top_city = bmw_city_sales_data.loc[bmw_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            bmw_city_sales['Year'].append(bmw_year)
            bmw_city_sales['City'].append(bmw_top_city['City'])
            bmw_city_sales['Number of Sales'].append(bmw_top_city['Number of Sales'])

        bmw_city_sales_df = pd.DataFrame(bmw_city_sales)  #create a DataFrame from the results
        st.table(bmw_city_sales_df)
        
    with tab8:
        toyota_data = electric_vehicles[electric_vehicles['Make'] == 'TOYOTA']  #filter the data for Toyota models
        toyota_top_model_data = toyota_data.groupby('Model').size().sort_values(ascending = False)  #find the top Toyota model
        top_toyota_model = toyota_top_model_data.index[0]
        top_toyota_model_data = toyota_data[toyota_data['Model'] == top_toyota_model]  #filter the data for the top model
        toyota_yearly_sales = top_toyota_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(toyota_yearly_sales.index, toyota_yearly_sales.values, color = '#00FFFF', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_toyota_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(toyota_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        toyota_data = electric_vehicles[electric_vehicles['Make'] == 'TOYOTA']  #filter the data for Toyota models
        toyota_top_model_data = toyota_data.groupby('Model').size().sort_values(ascending = False)  #find the top Toyota model
        top_toyota_model = toyota_top_model_data.index[0]
        top_toyota_model_data = toyota_data[toyota_data['Model'] == top_toyota_model]  #filter the data for the top model
        toyota_yearly_sales = top_toyota_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        toyota_sales_df = pd.DataFrame(toyota_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        toyota_sales_df.columns = ['Model Year', 'Units Sold']
        toyota_sales_df['Model Year'] = toyota_sales_df['Model Year'].astype(str)
        toyota_sales_df['Units Sold'] = toyota_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(toyota_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        toyota_years_of_interest = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  #define the years of interest
        toyota_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for toyota_year in toyota_years_of_interest:
            toyota_yearly_data = top_toyota_model_data[top_toyota_model_data['Model Year'] == toyota_year]  #filter data for the specific year
            toyota_city_sales_data = toyota_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            toyota_top_city = toyota_city_sales_data.loc[toyota_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            toyota_city_sales['Year'].append(toyota_year)
            toyota_city_sales['City'].append(toyota_top_city['City'])
            toyota_city_sales['Number of Sales'].append(toyota_top_city['Number of Sales'])

        toyota_city_sales_df = pd.DataFrame(toyota_city_sales)  #create a DataFrame from the results
        st.table(toyota_city_sales_df)
        
    with tab9: 
        volkswagen_data = electric_vehicles[electric_vehicles['Make'] == 'VOLKSWAGEN']  #filter the data for Volkswagen models
        volkswagen_top_model_data = volkswagen_data.groupby('Model').size().sort_values(ascending = False)  #find the top Volkswagen model
        top_volkswagen_model = volkswagen_top_model_data.index[0]
        top_volkswagen_model_data = volkswagen_data[volkswagen_data['Model'] == top_volkswagen_model]  #filter the data for the top model
        volkswagen_yearly_sales = top_volkswagen_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(volkswagen_yearly_sales.index, volkswagen_yearly_sales.values, color = '#FF7F50', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_volkswagen_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(volkswagen_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        volkswagen_data = electric_vehicles[electric_vehicles['Make'] == 'VOLKSWAGEN']  #filter the data for Volkswagen models
        volkswagen_top_model_data = volkswagen_data.groupby('Model').size().sort_values(ascending = False)  #find the top Volkswagen model
        top_volkswagen_model = volkswagen_top_model_data.index[0]
        top_volkswagen_model_data = volkswagen_data[volkswagen_data['Model'] == top_volkswagen_model]  #filter the data for the top model
        volkswagen_yearly_sales = top_volkswagen_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        volkswagen_sales_df = pd.DataFrame(volkswagen_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        volkswagen_sales_df.columns = ['Model Year', 'Units Sold']
        volkswagen_sales_df['Model Year'] = volkswagen_sales_df['Model Year'].astype(str)
        volkswagen_sales_df['Units Sold'] = volkswagen_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(volkswagen_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        volkswagen_years_of_interest = [2021, 2022, 2023, 2024]  #define the years of interest
        volkswagen_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for volkswagen_year in volkswagen_years_of_interest:
            volkswagen_yearly_data = top_volkswagen_model_data[top_volkswagen_model_data['Model Year'] == volkswagen_year]  #filter data for the specific year
            volkswagen_city_sales_data = volkswagen_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            volkswagen_top_city = volkswagen_city_sales_data.loc[volkswagen_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            volkswagen_city_sales['Year'].append(volkswagen_year)
            volkswagen_city_sales['City'].append(volkswagen_top_city['City'])
            volkswagen_city_sales['Number of Sales'].append(volkswagen_top_city['Number of Sales'])

        volkswagen_city_sales_df = pd.DataFrame(volkswagen_city_sales)  #create a DataFrame from the results
        st.table(volkswagen_city_sales_df)
                        
    with tab10: 
        jeep_data = electric_vehicles[electric_vehicles['Make'] == 'JEEP']  #filter the data for Jeep models
        jeep_top_model_data = jeep_data.groupby('Model').size().sort_values(ascending = False)  #find the top Jeep model
        top_jeep_model = jeep_top_model_data.index[0]
        top_jeep_model_data = jeep_data[jeep_data['Model'] == top_jeep_model]  #filter the data for the top model
        jeep_yearly_sales = top_jeep_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(jeep_yearly_sales.index, jeep_yearly_sales.values, color = '#008080', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_jeep_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(jeep_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        
        jeep_data = electric_vehicles[electric_vehicles['Make'] == 'JEEP']  #filter the data for Jeep models
        jeep_top_model_data = jeep_data.groupby('Model').size().sort_values(ascending = False)  #find the top Jeep model
        top_jeep_model = jeep_top_model_data.index[0]
        top_jeep_model_data = jeep_data[jeep_data['Model'] == top_jeep_model]  #filter the data for the top model
        jeep_yearly_sales = top_jeep_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        jeep_sales_df = pd.DataFrame(jeep_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        jeep_sales_df.columns = ['Model Year', 'Units Sold']
        jeep_sales_df['Model Year'] = jeep_sales_df['Model Year'].astype(str)
        jeep_sales_df['Units Sold'] = jeep_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(jeep_sales_df)

        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')
        
        jeep_years_of_interest = [2021, 2022, 2023, 2024]  #define the years of interest
        jeep_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for jeep_year in jeep_years_of_interest:
            jeep_yearly_data = top_jeep_model_data[top_jeep_model_data['Model Year'] == jeep_year]  #filter data for the specific year
            jeep_city_sales_data = jeep_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            jeep_top_city = jeep_city_sales_data.loc[jeep_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            jeep_city_sales['Year'].append(jeep_year)
            jeep_city_sales['City'].append(jeep_top_city['City'])
            jeep_city_sales['Number of Sales'].append(jeep_top_city['Number of Sales'])

        jeep_city_sales_df = pd.DataFrame(jeep_city_sales)  #create a DataFrame from the results
        st.table(jeep_city_sales_df)
        
    with tab11:
        hyundai_data = electric_vehicles[electric_vehicles['Make'] == 'HYUNDAI']  #filter the data for Hyundai models
        hyundai_top_model_data = hyundai_data.groupby('Model').size().sort_values(ascending = False)  #find the top Hyundai model
        top_hyundai_model = hyundai_top_model_data.index[0]
        top_hyundai_model_data = hyundai_data[hyundai_data['Model'] == top_hyundai_model]  #filter the data for the top model
        hyundai_yearly_sales = top_hyundai_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        
        #create a line chart
        plt.figure(figsize = (12, 6))
        plt.plot(hyundai_yearly_sales.index, hyundai_yearly_sales.values, color = '#B87333', linestyle = '-', marker = 'o', markersize = 4)
        plt.title(f'Number of {top_hyundai_model} sold per year', fontweight = 'bold', fontsize = 18, color = 'red')
        plt.xlabel('Model year', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.ylabel('Number of units sold', fontweight = 'bold', color = 'blue', fontsize = 12)
        plt.xticks(hyundai_yearly_sales.index, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True, linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        st.pyplot(plt.gcf())

        hyundai_data = electric_vehicles[electric_vehicles['Make'] == 'HYUNDAI']  #filter the data for Hyundai models
        hyundai_top_model_data = hyundai_data.groupby('Model').size().sort_values(ascending = False)  #find the top Hyundai model
        top_hyundai_model = hyundai_top_model_data.index[0]
        top_hyundai_model_data = hyundai_data[hyundai_data['Model'] == top_hyundai_model]  #filter the data for the top model
        hyundai_yearly_sales = top_hyundai_model_data.groupby('Model Year').size()  #count the number of units sold per year for the top model
        hyundai_sales_df = pd.DataFrame(hyundai_yearly_sales).reset_index()  #convert the series to a dataframe for clearer printing
        hyundai_sales_df.columns = ['Model Year', 'Units Sold']
        hyundai_sales_df['Model Year'] = hyundai_sales_df['Model Year'].astype(str)
        hyundai_sales_df['Units Sold'] = hyundai_sales_df['Units Sold'].apply(lambda x: f"{x:,}".replace(",", ""))
        st.table(hyundai_sales_df)
        
        st.write('The following will show the city, with the corresponding number of units sold, where the top model was best sold in the years of interest.')

        hyundai_years_of_interest = [2022, 2023, 2024]  #define the years of interest
        hyundai_city_sales = {'Year': [], 'City': [], 'Number of Sales': []}  #initialize a dictionary to store results

        for hyundai_year in hyundai_years_of_interest:
            hyundai_yearly_data = top_hyundai_model_data[top_hyundai_model_data['Model Year'] == hyundai_year]  #filter data for the specific year
            hyundai_city_sales_data = hyundai_yearly_data.groupby('City').size().reset_index(name = 'Number of Sales')  #group by city and count sales
            hyundai_top_city = hyundai_city_sales_data.loc[hyundai_city_sales_data['Number of Sales'].idxmax()]  #find the city with the highest sales
            
            #append results to the dictionary
            hyundai_city_sales['Year'].append(hyundai_year)
            hyundai_city_sales['City'].append(hyundai_top_city['City'])
            hyundai_city_sales['Number of Sales'].append(hyundai_top_city['Number of Sales'])

        hyundai_city_sales_df = pd.DataFrame(hyundai_city_sales)  #create a DataFrame from the results
        st.table(hyundai_city_sales_df)



##########correlation
elif current_tab == 'Correlation':
    st.title('Correlations between values')
    
    st.write('Now let''s see some correlations')
    
    #heatmap
    electric_vehicles = electric_vehicles.drop(['VIN (1-10)', 'Postal Code', 'Base MSRP', 'Legislative District', 'DOL Vehicle ID', 'Vehicle Location', '2020 Census Tract'], axis = 1)  #drop specified columns from the dataframe
    electric_vehicles_corr = electric_vehicles.corr(numeric_only = True)
    plt.figure(figsize = (12, 6))
    plt.figure(figsize = (12, 6))
    sns.heatmap(electric_vehicles_corr, annot = True, cmap = "RdYlBu", fmt = ".2f", linewidths = 0.5, square = True, cbar_kws = {"shrink": 0.8})
    plt.title('Correlation matrix of electric vehicles data', fontsize = 16, fontweight = 'bold', color = 'black')
    plt.xlabel('Features', fontsize = 12, fontweight = 'bold', color = 'green')
    plt.ylabel('Features', fontsize = 12, fontweight = 'bold', color = 'green')
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)

    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    filtered_data1 = electric_vehicles[(electric_vehicles['Model Year'] >= 2005) & (electric_vehicles['Electric Range'].notna())]  #filter the data to include only models from 2005 onwards and only with available electric range (not NaN)
    correlation1 = filtered_data1[['Model Year', 'Electric Range']].corr().iloc[0, 1]  #calculate the correlation between 'Model Year' and 'Electric Range'
    st.write(f'The correlation between model year (from 2005) and electric range is {correlation1:.2f}')
    
    filtered_data2 = electric_vehicles[(electric_vehicles['Model Year'] >= 2020) & (electric_vehicles['Model Year'] <= 2024) & (electric_vehicles['Electric Range'].notna())]  #filter the data to include only models from 2020 to 2025 and only with available electric range (not NaN)
    correlation2 = filtered_data2[['Model Year', 'Electric Range']].corr().iloc[0, 1]  #calculate the correlation between 'Model Year' and 'Electric Range'
    st.write(f'The correlation between model year and electric range (years 2021-2024) is {correlation2:.2f}')
    
    st.write('A negative correlation of -0.58 between ''Model Year'' and ''Electric Range'' related to the last 5 years, may seem weird, but several factors can explain it. Manufacturers may focus on features other than range, such as performance or connectivity. Diversification of models and possible discontinuation of those with high range may negatively affect the correlation.  In addition, a limited or unrepresentative data sample may skew the results. A drop in sales or market slowdown may lead manufacturers to reduce the supply of models with high autonomy. In summary, negative correlation is the result of various technological, strategic, and market factors.')
    
    st.write('Since the dataset has a large number of data, in order to have understandable graphs using linear regression, the data will be sampled at a fraction of 0.05%. In addition, this, also allows for a higher value negative correlation. ')
    
    vehicles_counts_by_name = electric_vehicles.groupby('Make').size().sort_values(ascending = False)
    top_companies = vehicles_counts_by_name.index[:10]
    
    filtered_data3 = electric_vehicles[
        (electric_vehicles['Model Year'] >= 2020) & 
        (electric_vehicles['Model Year'] <= 2024) & 
        (electric_vehicles['Electric Range'].notna()) &
        (electric_vehicles['Make'].isin(top_companies))
    ]  #filter the data to include only electric vehicles from model year 2020 to 2024, with a non-null electrice range and manufactured by one of the top companies 

    sampled_data = filtered_data3.sample(frac = 0.005, random_state = 42)  #reduce the data by sampling a fraction (0.005) of the original data
    correlation_sampled = sampled_data[['Model Year', 'Electric Range']].corr().iloc[0, 1]  #calculate the correlation between 'Model Year' and 'Electric Range' in the sampled data
    st.write(f'In fact, the correlation between (sampled) model year and electric range is {correlation_sampled:.2f}')
    


########## modelling with ML algorithms
elif current_tab == 'Modelling with ML Algorithms':
    st.title('Modelling with Machine Learning Algorithms')
    
    vehicles_counts_by_name = electric_vehicles.groupby('Make').size().sort_values(ascending = False)
    top_companies = vehicles_counts_by_name.index[:10]
    filtered_data3 = electric_vehicles[
        (electric_vehicles['Model Year'] >= 2020) & 
        (electric_vehicles['Model Year'] <= 2024) & 
        (electric_vehicles['Electric Range'].notna()) &
        (electric_vehicles['Make'].isin(top_companies))
    ]  #filter the data to include only electric vehicles from model year 2020 to 2024, with a non-null electrice range and manufactured by one of the top companies 
    sampled_data = filtered_data3.sample(frac = 0.005, random_state = 42)  #reduce the data by sampling a fraction (0.005) of the original data
    correlation_sampled = sampled_data[['Model Year', 'Electric Range']].corr().iloc[0, 1]  #calculate the correlation between 'Model Year' and 'Electric Range' in the sampled data
    x = sampled_data['Model Year'].to_numpy().reshape(-1, 1)  #convert the 'Model Year' column to a numpy array and reshape it to have one column
    y = sampled_data['Electric Range'].to_numpy()  #convert the 'Electric Range' column to a numpy array and reshape it to have one column
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 0)  #split the dataset into training and testing sets
    
    ro_scaler = RobustScaler()  #RobustScaler is used to scale features by removing the median and scaling according to the quantile range
    x_train_scaled = ro_scaler.fit_transform(x_train)  #fit the scaler on the training data and transform it
    x_test_scaled = ro_scaler.fit_transform(x_test)  #apply the same transformation to the test data
    
    model = LinearRegression()  #LinearRegression is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (features)
    model.fit(x_train_scaled, y_train)  #train the LinearRegression model using the scaled training data
    
    y_pred = model.predict(x_test_scaled)  #the .predict() method computes the predicted values of the target variable based on the test features (x_test_scaled)
    
    st.write('Now let''s see some ML features.')

    score = model.score(x_test_scaled, y_test)  #the .score() method computes how well the model's predictions match the actual target values
    st.write("Coefficient of determination (R-squared):", score)
    
    st.write('The R-squared score ranges from 0 to 1, where 1 indicates a perfect fit and 0 indicates no explanatory power. An R-squared value of -0.06721275511743419 indicates that the model is performing poorly, even worse than a simple model that predicts the average target value.')
    
    slope = model.coef_  #the .coef_ attribute of the LinearRegression model contains the coefficients (weights) for each feature
    st.write('Coefficients:', slope)
    
    st.write('The coefficient value of -82.08462005 indicates a strong negative relationship between the feature and the target variable. For every one-unit increase in the feature, the target variable is predicted to decrease by approximately 82.08 units. ')
    
    intercept = model.intercept_  #the .intercept_ attribute contains the intercept of the linear regression equation
    st.write('Intercepts:', intercept)
    
    st.write('The intercept value of -3.460796075479035 indicates that, when all feature values are zero, the model predicts a target value of approximately -3.46. This negative intercept represents the baseline prediction of the target variable and helps to set the starting point for the model''s predictions before the influence of the features is considered.')
    
    r2 = r2_score(y_test, y_pred)  # The r2_score function computes the R-squared value, which measures how well the model's predictions match the actual target values
    st.write('Coefficient of determination (R^2):', r2)
    
    st.write('The R-squared value of -0.06721275511743419 indicates that the model''s performance is poor and worse than predicting the mean of the target variable. This negative value suggests that the model is not capturing the underlying patterns in the data effectively, and there may be issues with the model choice, feature selection, or data quality. ')
    
    mse = mean_squared_error(y_test, y_pred)  #the mean_squared_error function computes the average of the squared differences between the actual and predicted values
    target_variance = np.var(y_test)  #np.var computes the variance of the actual target values
    mse_to_variance_ratio = mse / target_variance  #this ratio provides a measure of how the model's error compares to the variance in the target variable
    st.write("MSE to Variance Ratio:", mse_to_variance_ratio)
    
    st.write('A MSE to Variance Ratio of 1.0672127551174342 indicates that the Mean Squared Error of the model''s predictions is slightly larger than the variance of the target variable. This suggests that the model’s errors are significant relative to the natural variability in the target, and the model may not be performing well. ')
    
    plt.figure(figsize = (10, 6))
    plt.plot(y_test, color = 'blue', linewidth = 1.5, label = 'Actual')
    plt.plot(y_pred, linestyle = '--', color = 'red', linewidth = 1, label = 'Predicted')
    plt.xlabel('Index', fontsize = 12, fontweight = 'bold', color = 'black')
    plt.ylabel('Values', fontsize = 12, fontweight = 'bold', color = 'black')
    plt.title('Comparison of Actual vs. Predicted Values', fontsize = 15, fontweight = 'bold', color = 'black')
    plt.xticks(fontsize = 10, color = 'black')
    plt.yticks(fontsize = 10, color = 'black')
    plt.grid(True, linestyle = '--', alpha = 0.5)
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    st.write('Already from the numerical values it could be expected that the graph would be inaccurate. Here it can actually be seen that the actual values are quite discordant with the predicted values.')
    
    intercept = model.intercept_  #he .intercept_ attribute contains the constant term of the linear regression equation
    slope = model.coef_[0]  # The .coef_ attribute contains the coefficient for each feature in the model
    st.write("Regression line equation:")
    st.write('Y = {:.2f} + {:.3f} * X'.format(intercept, slope))
    
    plt.figure(figsize=(10, 6))
    plt.title('Training and Test Plot with Prediction Line', fontsize = 16, fontweight = 'bold', color = 'black')
    plt.scatter(x_train, y_train, label = 'Training points', color = 'red', marker = 'o', s = 50, alpha = 0.7)
    plt.scatter(x_test, y_test, label = 'Test points', color = 'blue', marker = 's', s = 50, alpha = 0.7)
    plt.plot(x_test, y_pred, label = 'Prediction line', color = 'green', linewidth = 2)
    plt.grid(True, linestyle = '--', alpha = 0.5)
    plt.xlabel('X values', fontsize = 14, fontweight = 'bold', color = 'black')
    plt.ylabel('Y values', fontsize = 14, fontweight = 'bold', color = 'black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    residuals = y_test - y_pred  #esiduals are the differences between the actual target values (y_test) and the predicted values (y_pred) and they represent the errors or deviations of the model's predictions from the true values
    plt.figure(figsize = (10,6))
    sns.histplot(residuals, kde = True, color = 'skyblue')
    plt.title('Histogram of residuals', fontsize = 16, fontweight = 'bold', color='red')
    plt.xlabel('Residuals', fontsize = 12, fontweight = 'bold', color = 'black')
    plt.ylabel('Frequency', fontsize = 12, fontweight = 'bold', color = 'black')
    plt.xticks(fontsize = 10, color = 'black')
    plt.yticks(fontsize = 10, color = 'black')
    plt.grid(True, linestyle = '--', alpha = 0.5)

    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    numeric_columns = ['Model Year', 'Electric Range']  #selects columns that contain numerical data for scaling
    numeric_data = sampled_data[numeric_columns]

    scaler = StandardScaler()  #StandardScaler standardizes features by removing the mean and scaling to unit variance
    scaled_numeric_data = scaler.fit_transform(numeric_data)

    scaled_data = pd.DataFrame(scaled_numeric_data, columns = numeric_columns, index = sampled_data.index)  #create a dataframe with the scaled numeric data, using the original column names and index from sampled_data
    
    kmeans = KMeans(n_clusters = 10, random_state = 0, n_init = 10)  #initialize the KMeans clustering algorithm
    kmeans.fit(scaled_data)  #the .fit method computes the K-means clustering on the scaled data
    
    cluster_labels = kmeans.labels_  #the .labels_ attribute contains the cluster labels for each data point in the dataset
    
    plt.figure(figsize = (10, 6))
    plt.scatter(scaled_data['Model Year'], scaled_data['Electric Range'], c = cluster_labels, cmap = 'viridis', s = 50, alpha = 0.5)
    plt.xlabel('CO Mean', fontweight = 'bold', fontsize = 14, color = 'black')
    plt.ylabel('NO2 Mean', fontweight = 'bold', fontsize = 14, color = 'black')
    plt.title('CO Mean - NO2 Mean Clusters', fontweight = 'bold', fontsize = 18, color = 'red')
    plt.colorbar(label = 'Cluster')

    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    st.write('The graph shows the relationship between the year of production and the electric range of electric vehicles, grouped into different clusters identified by different colors. The dots in the graph indicate vehicle models that have been grouped into distinct clusters based on their range characteristics. A clear trend can be seen in the data, where newer vehicles tend to be clustered together (right end), suggesting that newer models have similar range characteristics. In addition, there are areas of the graph where data are missing, which could indicate an absence of vehicle models in those years or autonomy ranges, suggesting potential gaps in the market or areas for future development.')