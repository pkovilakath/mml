Compare multiple machine learning classification algorithms and select the best AI solution for your data. Select a training file and set the appropriate initial conditions to get going.

This application is written in Python and  utilizes the Django based web framework to allow for easy attractive web functionality.  The application depends heavily on the Scikit-learn machine learning library.

Once the mml web app is running, the **learn module**, allows users to upload a csv file with the data they need to analyze, check the appropriate checkboxes to compare and contrast different machine learning algorithms.  

The users then save out the models they think show promise.  The user then evokes the **predict module** to test new data (or old).  This is done by specifying the new csv data file and then selecting the module saved from  the learn module.

Users may find this  application useful to make a better selection of the machine learning algorithm they want to focus and drill down on.

## Installation

 1. Clone this git project 
 2. Make sure you have Python 3 
 3. Start the project  with **python manage.py runserver** or from the ./vmml directory, **bin/python3 mml/manage.py runserver**
 4. Point you browser at the application (assuming a local install)
    **http://127.0.0.1:8000/learn**
