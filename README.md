# Credit-Card-Fraud
Detecting credit card fraud using simulated tabular data.

This project was made for the course AMS 561 at stonybrook university. Its purpose was to create a system using machine learning to detect credit card fraud in real time. 

Target: 
- 1 (Fraudulent Transaction)
- 0 (Non Fraudulent Transaction)

The functionality of the project is as follows:
- Data is stored virtually on Google Cloud Bucket accesible using SSH key criterion
- A virtual machine or notebook ran by google uses the driver file to load in the training and testing data from the bucket using utils.py
- Data cleaning and handling is performed via data_transformation.py
- Model training is then performed by the model_training utilities
- The final model is saved and uploaded back to the cloud bucket for later use in a pickle file format
- All details regarding transformations, training and saving were done via the custom exception class and the logger, driven by the driver file.

# Techniques Used
**Data Transformation:** When transforming data: numpy, pandas and Scikit-Learn is utilized to remove sensitive data like names and address, while dealing with numeric and categorical data through two seperate pipilines in the DataTransformation class. Numerical features were standardized and categorical features were one hot encoded with unknowns being represented as 0's across all variables.

**Model Training:** Model training was done with cross validation across three different model types: Logistic Regression, XGBoost and SVM Classifier. These models each had their own grid search parameters that were used and scored based upon ROC AUC using the test set. The best of the three variants is then saved and returned. 

**Utils:** This file contains the functions used for loading/uploading the cloud utilities and pickle file conversion.

**Driver:** Driver runs all the core functionality from loading the data, transforming, training and eventually uploading the best model, all while logging the steps along the way with diagnostics for bug checking.

# Results and Challenges
The model was succesful in reducing credit card fraud accoridng to its confusion matrix. If implemented the estimates in reducing fraud were around 60%, with a low amount of false positives. A interesting finding of this project is that there is no linear seperability among classes. This model was hosted very briefly on Google Cloud for presenting, but there were difficulties in using json as a format to input values.

There were many notable challenges in creating this project like pushing the model to production, creating all the systems to make this project work off a virtual machine and understanding key components. As I did not check the time complexity of the SVM Classifier I ended up waisting many hours of training on a unusable model in the long term. 

While I did not intend to use github as a way to download and run the code via command line prompts on a virtual enviroment originally, it proved to be insanely useful but at the same time took a while to learn. 
