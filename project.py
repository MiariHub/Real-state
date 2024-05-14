import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import joblib


st.set_page_config(page_title="RealEstate!!!", page_icon=":bar_chart:",layout="wide")

st.markdown("<h1 style='text-align: center; color: white;font-family: Times New Roman'>House Price Prediction Using AI</h1>", unsafe_allow_html=True)


#st.title("House Price Prediction Using AI")
# Load the dataset

df = pd.read_csv("Real estate valuation data set.csv")



menu = st.sidebar.radio("Menu", ["Home", "Dashboard","Prediction","About"])
if menu == "Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(' ')
    
    with col2:
        st.image("pngegg (8).png", caption="Real Estate Prediction")
    
    with col3:
        st.write(' ')

    st.markdown("""
    <div style="text-align: justify;"font-family: Times New Roman>
        Welcome to our state-of-the-art platform designed to predict real estate prices with precision. 
        Leveraging advanced machine learning techniques, particularly Linear Regression, we analyze 
        comprehensive datasets sourced from Sindian District, New Taipei City, Taiwan. By considering 
        key attributes such as transaction date, house age, proximity to amenities like transportation 
        hubs and convenience stores, as well as geographic coordinates, our model provides accurate 
        predictions of property prices. This predictive capability enables stakeholders in the real 
        estate market to make informed decisions, whether it's buying, selling, or investing in properties. 
        Join us as we redefine the future of real estate valuation, making property pricing more transparent, 
        efficient, and reliable.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: Right;">By Developer.</div>', unsafe_allow_html=True)
if menu == "About":
    st.write("""
    ### About the Application
    This application predicts real estate prices using advanced machine learning techniques.
    Here you'll find detailed information on how to use the app, the data behind it, and the methods applied for price prediction.
    #### How to Use
    - **Home**: Get a brief overview and introduction to our platform.
    - **Dashboard**: Visualize different aspects of the real estate data through interactive charts and graphs.
    - **Prediction**: Enter specific house features to receive a real-time price prediction.
    #### Data Used
    The data used in this application comes from the Sindian District, New Taipei City, Taiwan. It includes several features like transaction date, house age, distance to the nearest MRT station, number of convenience stores nearby, latitude, and longitude.
    #### Prediction Methodology
    The application uses a Linear Regression model trained on historical data to provide price predictions. The model takes into account various attributes of properties to make its predictions as accurate as possible.
    """)

if menu == "Dashboard":
    n_rows = st.slider("Choose number of rowes to display " , min_value=5, max_value= len(df), step=1)
    columns_to_show= st.multiselect("Select columns to show", df.columns.to_list(), default= df.columns.to_list())
    st.write(df[:n_rows][columns_to_show])
    col1, col2, col3  = st.columns(3)

    numarical_columns = df.select_dtypes(include=np.number).columns.to_list()
    tab1, tab2, tab3 = st.tabs(["Scatter plot", "Histogram","Descriptive Statistics"])
   
    
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

      
       

    with tab3:
        # User options to select data presentation
       st.write(df.describe())
       analysis_type = st.selectbox("Select Analysis Type", ["Box Plot", "Correlation Matrix"])
    
       if analysis_type == "Box Plot":
            # Allowing user to select a column for box plot
            column_to_plot = st.selectbox("Select Column for Box Plot", df.columns)
            fig = px.box(df, y=column_to_plot)
            st.plotly_chart(fig)

        # Compute correlation matrix
       elif analysis_type == "Correlation Matrix":
            corr_matrix = df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", labels=dict(color="Correlation"),
                            x=corr_matrix.columns, y=corr_matrix.columns,
                            color_continuous_scale='RdBu_r', origin='lower')
            fig.update_xaxes(side="bottom")
            st.plotly_chart(fig)

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
st.sidebar.write(" This app is continuously being improved and updated following Agile practices.")    


       

