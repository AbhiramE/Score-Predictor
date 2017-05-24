The overall layout of the project is broken down into packages according to their task.

We have 3 packages,
1. Collection
2. Evaluation
3. Learning

1. Collection
    -> The collection folder contains the code for scarping the data from the site.
    -> Scraper contains all the scaper functions for specific types of data, like ground average, team average
    etc.
    -> data_file_cleaner cleans the ground averages scraped and adds that to csv file.
    -> convertor.py This file builds the dataset with the required features for first innings and second innings.

2. Learning
    This folder contains all the model trained.

    1st innings
    This package contains the pipelines implemented for all model trained for first innings scenario.
    It takes the training file, runs the hyperparameter optimization on it and fits the model with the
    best parameters obtained. It then saves the best fit model as a pickle.

    2nd innings
    This package contains the pipelines implemented for all model trained for second innings scenario.
    It takes the training file, runs the hyperparameter optimization on it and fits the model with the
    best parameters obtained. It then saves the best fit model as a pickle.

3. Evaluation
   The evaluation folder contains evaluation.py which contains the final prediction on the test data.
   Edit the link to the pickle to evaluate different models. Then execute the file to obtain the ML RMSE
   and DL RMSE. The evaluation files for the each innings is contained in the respective folders.

Running a file:
1. All files in Learning are runnable provided the the datasets are obtained.
2. All files in Evaluation are runnable provided respective pickles are available for the files.
3. The scaper files can be run my calling the functions for respective features.
