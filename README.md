# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python nonstandardcode.py
# To create an environment 
conda env --name mle-dev biopython
conda activate mle-dev
#To export the environment to env.yml
conda env export mle-dev > env.yml
# mle-training
# assignment-2.1(coding best  practices)
-intially i dived the code into three files ingest_data.py,score.py,train.py
-in ingest_data.py we have function in which we take the housing data from the git account and store the data and formatt it
-in train.py i have divded my housing data into train and test sets and now based on that  i have trained it using above three techniques.
-in score.py we will analyze the data and get the score and predicted values and addition to that i have added logging status beacause to know the status of the program while running and set log level also
-we have log files in that we will have output of our programme
-in tests folder there are unit and functional testing files for doing basic tests on the codes.
# making project  into a package
-In sklearn pypi create a token and with that token we can pack our project into a set of installable package.
-now we should create and activate the package by doing this commands
-pip install -e .
-pip install build
-python -m build
-conda install twine
-twine upload dist/*
-after these commands our project will be like packages we can install it directly
# Creating html files using sphinx
-we go to sphinx and cretae the html files of our project.
# installing the package
-pip install housing_price_prediction
# testing in our system
1.create a folder with dist and env files
2.run using python -m housing_price_prediction.test
