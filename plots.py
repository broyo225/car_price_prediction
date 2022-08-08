# Code for 'main_app.py' file.
# Importing the necessary Python modules.
import streamlit as st
import numpy as np
import pandas as pd

# Import the individual Python files
import home
import data
import plots
import predict

# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title = 'Car Price Prediction',
                    page_icon = ':car:',
                    layout = 'centered',
                    initial_sidebar_state = 'auto'
                    )

# Dictionary containing positive integers in the form of words as keys and corresponding former as values.
words_dict = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}
def num_map(series):
    return series.map(words_dict)

# Loading the dataset.
@st.cache()
def load_data():
    # Reading the dataset
    cars_df = pd.read_csv("car-prices.csv")
    # Extract the name of the manufactures from the car names and display the first 25 cars to verify whether names are extracted successfully.
    car_companies = pd.Series([car.split(" ")[0] for car in cars_df['CarName']], index = cars_df.index)
    # Create a new column named 'car_company'. It should store the company names of a the cars.
    cars_df['car_company'] = car_companies
    # Replace the misspelled 'car_company' names with their correct names.
    cars_df.loc[(cars_df['car_company'] == "vw") | (cars_df['car_company'] == "vokswagen"), 'car_company'] = 'volkswagen'
    cars_df.loc[cars_df['car_company'] == "porcshce", 'car_company'] = 'porsche'
    cars_df.loc[cars_df['car_company'] == "toyouta", 'car_company'] = 'toyota'
    cars_df.loc[cars_df['car_company'] == "Nissan", 'car_company'] = 'nissan'
    cars_df.loc[cars_df['car_company'] == "maxda", 'car_company'] = 'mazda'
    cars_df.drop(columns= ['CarName'], axis = 1, inplace = True)
    cars_numeric_df = cars_df.select_dtypes(include = ['int64', 'float64']) 
    cars_numeric_df.drop(columns = ['car_ID'], axis = 1, inplace = True)
    # Map the values of the 'doornumber' and 'cylindernumber' columns to their corresponding numeric values.
    cars_df[['cylindernumber', 'doornumber']] = cars_df[['cylindernumber', 'doornumber']].apply(num_map, axis = 1)
    # Create dummy variables for the 'carbody' columns.
    car_body_dummies = pd.get_dummies(cars_df['carbody'], dtype = int)
    # Create dummy variables for the 'carbody' columns with 1 column less.
    car_body_new_dummies = pd.get_dummies(cars_df['carbody'], drop_first = True, dtype = int)
    # Create a DataFrame containing all the non-numeric type features.
    cars_categorical_df = cars_df.select_dtypes(include = ['object'])
    #Get dummy variables for all the categorical type columns using the dummy coding process.
    cars_dummies_df = pd.get_dummies(cars_categorical_df, drop_first = True, dtype = int)
    #  Drop the categorical type columns from the 'cars_df' DataFrame.
    cars_df.drop(list(cars_categorical_df.columns), axis = 1, inplace = True)
    # Concatenate the 'cars_df' and 'cars_dummies_df' DataFrames.
    cars_df = pd.concat([cars_df, cars_dummies_df], axis = 1)
    #  Drop the 'car_ID' column
    cars_df.drop('car_ID', axis = 1, inplace = True)
    final_columns = ['carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick', 'price']
    return cars_df[final_columns]
  
final_cars_df = load_data()

# Adding a navigation in the sidebar using radio buttons.
# Create a dictionary.
pages_dict = {
                "Home": home,
                "View Data": data, 
                "Visualise Data": plots, 
                "Predict": predict
            }

# Add radio buttons in the sidebar for navigation and call the respective pages based on user selection.
st.sidebar.title('Navigation')
user_choice = st.sidebar.radio("Go to", tuple(pages_dict.keys()))
if user_choice == "Home":
    home.app() 
else:
    selected_page = pages_dict[user_choice]
    selected_page.app(final_cars_df) 
# This 'app()' function is defined in all the 'home.py', data.py', 'plots.py' and 'predict.py' files. 
# Whichever option out of "View Data", "Visualise Data" and "Predict" a user selects, that option gets stored in the 
# 'selection' variable and the correspoding value to that key gets stored in the 'page' variable and then the 'app()'
# function gets called from that Python file 
# which could be either of 'home.py', 'data.py', 'plots.py' and 'predict.py' files in this case.
##############################################
# Code for 'home.py' file.
import streamlit as st

def app():
    st.header("Car Price Prediction App")
    st.text("""
            This web app allows a user to predict the prices of a car based on their 
            engine size, horse power, dimensions and the drive wheel type parameters.
            """
            )
#################################################
# Code for 'data.py' file
# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st

# Define a function 'app()' which accepts 'car_df' as an input.
def app(car_df):
    st.header("View Data")
    # Add an expander and display the dataset as a static table within the expander.
    with st.beta_expander("View Dataset"):
        st.table(car_df)

    st.subheader("Columns Description:")
    if st.checkbox("Show summary"):
        st.table(car_df.describe())

    beta_col1, beta_col2, beta_col3 = st.beta_columns(3)

    # Add a checkbox in the first column. Display the column names of 'car_df' on the click of checkbox.
    with beta_col1:
        if st.checkbox("Show all column names"):
            st.table(list(car_df.columns))

    # Add a checkbox in the second column. Display the column data-types of 'car_df' on the click of checkbox.
    with beta_col2:
        if st.checkbox("View column data-type"):
            st.table(car_df.dtypes)

    # Add a checkbox in the third column followed by a selectbox which accepts the column name whose data needs to be displayed.
    with beta_col3:
        if st.checkbox("View column data"):
            column_data = st.selectbox('Select column', tuple(car_df.columns))
            st.write(car_df[column_data])

##################################################
# S1.1: Design the "Visualise Data" page of the multipage app.
# Import necessary modules 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function 'app()' which accepts 'car_df' as an input.
def app(car_df):
    st.header('Visualise data')
    # Remove deprecation warning.
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Subheader for scatter plot.
    st.subheader("Scatter plot")
    # Choosing x-axis values for scatter plots.
    features_list = st.multiselect("Select the x-axis values:", 
                                            ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
    # Create scatter plots.
    for feature in features_list:
        st.subheader(f"Scatter plot between {feature} and price")
        plt.figure(figsize = (12, 6))
        sns.scatterplot(x = feature, y = 'price', data = car_df)
        st.pyplot()

    # Add a multiselect widget to allow the user to select multiple visualisation.
    # Add a subheader in the sidebar with label "Visualisation Selector"
    st.subheader("Visualisation Selector")

    # Add a multiselect in the sidebar with label 'Select the charts or plots:'
    # and pass the remaining 3 plot types as a tuple i.e. ('Histogram', 'Box Plot', 'Correlation Heatmap').
    # Store the current value of this widget in a variable 'plot_types'.
    plot_types = st.multiselect("Select charts or plots:", ('Histogram', 'Box Plot', 'Correlation Heatmap'))

    # Display box plot using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.
    if 'Histogram' in plot_types:
        st.subheader("Histogram")
        columns = st.selectbox("Select the column to create its histogram",
                                      ('carwidth', 'enginesize', 'horsepower'))
        # Note: Histogram is generally created for continuous values not for discrete values.
        plt.figure(figsize = (12, 6))
        plt.title(f"Histogram for {columns}")
        plt.hist(car_df[columns], bins = 'sturges', edgecolor = 'black')
        st.pyplot()

    # Create box plot using the 'seaborn' module and the 'st.pyplot()' function.
    if 'Box Plot' in plot_types:
        st.subheader("Box Plot")
        columns = st.selectbox("Select the column to create its box plot",
                                      ('carwidth', 'enginesize', 'horsepower'))
        plt.figure(figsize = (12, 2))
        plt.title(f"Box plot for {columns}")
        sns.boxplot(car_df[columns])
        st.pyplot()

    # Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
    if 'Correlation Heatmap' in plot_types:
        st.subheader("Correlation Heatmap")
        plt.figure(figsize = (8, 5))
        ax = sns.heatmap(car_df.corr(), annot = True) # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim() # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5) # Increasing the bottom and decreasing the bottom margins respectively.
        st.pyplot()
################################################
# S2.1: Import the necessary Python modules and create the 'prediction()' function as directed above.
# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

# Define the 'prediction()' function.
@st.cache()
def prediction(car_df, car_width, engine_size, horse_power, drive_wheel_fwd, car_comp_buick):
    X = car_df.iloc[:, :-1] 
    y = car_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    score = lin_reg.score(X_train, y_train)

    price = lin_reg.predict([[car_width, engine_size, horse_power, drive_wheel_fwd, car_comp_buick]])
    price = price[0]

    y_test_pred = lin_reg.predict(X_test)
    test_r2_score = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_msle = mean_squared_log_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return price, score, test_r2_score, test_mae, test_msle, test_rmse
######################################################
# S2.2: Define the 'app()' function as directed above.
def app(car_df): 
    st.markdown("<p style='color:blue;font-size:25px'>This app uses <b>Linear regression</b> to predict the price of a car based on your inputs.", unsafe_allow_html = True) 
    st.subheader("Select Values:")
    car_wid = st.slider("Car Width", float(car_df["carwidth"].min()), float(car_df["carwidth"].max()))     
    eng_siz = st.slider("Engine Size", int(car_df["enginesize"].min()), int(car_df["enginesize"].max()))
    hor_pow = st.slider("Horse Power", int(car_df["horsepower"].min()), int(car_df["horsepower"].max()))    
    drw_fwd = st.radio("Is it a forward drive wheel car?", ("Yes", "No"))
    if drw_fwd == 'No':
        drw_fwd = 0
    else:
        drw_fwd = 1
    com_bui = st.radio("Is the car manufactured by Buick?", ("Yes", "No"))
    if com_bui == 'No':
        com_bui = 0
    else:
        com_bui = 1
    
    # When 'Predict' button is clicked, the 'prediction()' function must be called 
    # and the value returned by it must be stored in a variable, say 'price'. 
    # Print the value of 'price' and 'score' variable using the 'st.success()' and 'st.info()' functions respectively.
    if st.button("Predict"):
        st.subheader("Prediction results:")
        price, score, car_r2, car_mae, car_msle, car_rmse = prediction(car_df, car_wid, eng_siz, hor_pow, drw_fwd, com_bui)
        st.success("The predicted price of the car: ${:,}".format(int(price)))
        st.info("Accuracy score of this model is: {:2.2%}".format(score))
        st.info(f"R-squared score of this model is: {car_r2:.3f}")  
        st.info(f"Mean absolute error of this model is: {car_mae:.3f}")  
        st.info(f"Mean squared log error of this model is: {car_msle:.3f}")  
        st.info(f"Root mean squared error of this model is: {car_rmse:.3f}")
