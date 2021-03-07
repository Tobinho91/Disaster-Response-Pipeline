
# Disaster Response Pipeline Project

## Table of contents

- [Motivation](#Motivation)
- [Quick start](#Quick-start)
- [File description](#File-description)
- [How to interact](#How-to-interact)
- [Summary](#Summary)
- [Versioning](#Versioning)
- [Creator](#Creator)
- [Thanks](#Thanks)
- [Copyright and license](#Copyright-and-license)

# Motivation

## Udacity Data Science Project

The default branch is for development of my third Projet release for the Udacity nanodegree program.
It includes a ETL pipeline and a ML pipeline combined with a web app to display the results.

The project is about a "Disaster Response Pipeline Project"

This project classifies disaster response messages through machine learning.


# Quick start

The project applies skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. Additionally, it shows three visualizations for the end-user on a web app.
It inlcudes front- and back-end development of the data and its set-up.


# File description

### Data

- process_data.py: reads in the data, cleans and stores it in a SQL database. Basic usage is python process_data.py MESSAGES_DATA CATEGORIES_DATA NAME_FOR_DATABASE
- categories.csv and messages.csv (dataset)
- DisasterResponse.db: created database from transformed and cleaned data.

### Models
- train_classifier.py: includes the code necessary to load the data, transform it using natural language processing, run a machine learning model using GridSearchCV and train it. Basic usage is python train_classifier.py DATABASE_DIRECTORY SAVENAME_FOR_MODEL

### App
- run.py: Flask app and the user interface used to predict results and display them.
- wrangling_graphs.py: includes the visualizations based with plotly in a separate script 
- templates: folder containing the html templates



# How to interact

- Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans the data and stores in database:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 

- To run ML pipeline that trains classifier and saves it:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 

- Run the following command in the app's directory to run your web app:

python run.py

- Go to http://0.0.0.0:3001/

# Summary

## Screenshots


### First page overview

![PNG_1.PNG](attachment:PNG_1.PNG)


### Shows the output after inserting a text message in the "Result" section below

![PNG_2.PNG](attachment:PNG_2.PNG)

### The whole first page including three customized visualizations

![PNG_3.PNG](attachment:PNG_3.PNG)


# Versioning

The current version is 1.0

# Creator

**Tobias Petri**

# Thanks

**To the following links/pages:**

- All the information provided by Udacity: www.udacity.com
- Stack overflow - Where Developers Learn, Share, & Build ...: www.stackoverflow.com
- Python documentation: www.python.org

# Copyright and license

Code and documentation copyright 2021–2021 by **Tobias Petri**

