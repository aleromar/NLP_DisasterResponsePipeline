# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project Summary:
The project is divided in three main modules: an ETL pipeline, an ML pipeline and an app to showcase the results

The ETL pipeline reads information from two files and after doing some cleaning and processing it creates an SQL database to store the results
    - process_data.py is the main file for this module that contains code that accomplish the goal described above
    - input files are also stored in the same folder. These include important features that allow to create a ML model later on
The ML pipeline reads the data resulting from running the previous pipeline and builds a machine learning model that makes predictions on unseen data.
    - train_classifier.py is the main file for this module and contains code to meet the requirement above
    - after running the ML pipeline the model file should be located under the same folder
The app is able to make predictions on unseen data and also creates some graphs that show properties of the data under analysis.
    - run.py is the main file for this module and contains all the relevant code used for make predictions and create graphs