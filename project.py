import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import joblib


st.set_page_config(page_title="RealEstate!!!", page_icon=":bar_chart:",layout="wide")



st.title("House Price Prediction Using AI")
# Load the dataset

df = pd.read_csv("Real estate valuation data set.csv")



menu = st.sidebar.radio("Menu", ["Home", "Dashboard","Prediction"])

if menu == "Home":
   st.image("pngegg (8).png", caption="Real Estate Prediction")
   st.write("""
    Welcome to our state-of-the-art platform designed to predict real estate prices with precision. 
    Leveraging advanced machine learning techniques, particularly Linear Regression, we analyze 
    comprehensive datasets sourced from Sindian District, New Taipei City, Taiwan. By considering 
    key attributes such as transaction date, house age, proximity to amenities like transportation 
    hubs and convenience stores, as well as geographic coordinates, our model provides accurate 
    predictions of property prices. This predictive capability enables stakeholders in the real 
    estate market to make informed decisions, whether it's buying, selling, or investing in properties. 
    Join us as we redefine the future of real estate valuation, making property pricing more transparent, 
    efficient, and reliable.
   """)

if menu == "Dashboard":
    n_rows = st.slider("Choose numbor of rowes to display " , min_value=5, max_value= len(df), step=1)
    columns_to_show= st.multiselect("Select columns to show", df.columns.to_list(), default= df.columns.to_list())
    st.write(df[:n_rows][columns_to_show])
    col1, col2, col3  = st.columns(3)

    numarical_columns = df.select_dtypes(include=np.number).columns.to_list()
    tab1, tab2 = st.tabs(["Scatter plot", "Histogram"])
   
    
    with tab1 : 
       with col1: 
          x_column = st.selectbox("Select column on X_axis:", numarical_columns)

       with col2:
          y_column = st.selectbox("Select column on y_axis:", numarical_columns)

       with col3:
          color = st.selectbox("Select column to be color:", numarical_columns)

       fig_scatter = px.scatter(df, x= x_column, y= y_column , color=color)
       st.plotly_chart(fig_scatter)

    with tab2:
       histogram_featuer = st.selectbox("Selsect feature to histogram", numarical_columns)
       fig_hist = px.histogram(df, x=histogram_featuer ) 
       st.plotly_chart(fig_hist)
if menu == "Prediction":
    
    model_filename = "linear_model.pkl"
    loaded_model = joblib.load(model_filename)

    
    st.title("House Price Prediction")

    
    st.header("Enter House Features")
    transaction_date = st.number_input("Transaction Date", value=2013)
    house_age = st.number_input("House Age (Years)", value=10)
    distance_to_mrt = st.number_input("Distance to Nearest MRT Station (meters)", value=500)
    num_convenience_stores = st.number_input("Number of Convenience Stores", value=5)
    latitude = st.number_input("Latitude", value=24.971)
    longitude = st.number_input("Longitude", value=121.540)

    
    input_features = [[transaction_date, house_age, distance_to_mrt, num_convenience_stores, latitude, longitude]]
    predicted_price = loaded_model.predict(input_features)

    
    st.subheader("Predicted House Price")
    st.write(f"The predicted price of the house is {predicted_price[0]:.2f} New Taiwan Dollar/Ping.")
   

       

