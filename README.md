# Disaster Response Pipeline Project
![Disaster Response](./img/disaster-response.jpg)
This repository contains a fullstack web app and an ETL (Extract Transform Learning) and machine
learning pipeline from the disaster response dataset provided by FigureEight (now Appen) which
could be accessed here: [https://appen.com/datasets/combined-disaster-response-data/](https://appen.com/datasets/combined-disaster-response-data/).

A view of the sourcecode could be accessed on GitHub here: [https://github.com/WisnuMulya/Disaster-Response-Project](https://github.com/WisnuMulya/Disaster-Response-Project).

## Installation ##
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
		To run GridSearch, pass the argument of `grid_search=True` on `line 206` under `train_classifier.py`.
		

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Remarks ##
* On the data
  There is an inconsistency in the values of the `related` category: the values are
  `0`, `1`, and `2`. The approach I took to deal with this is that I change the 
  value of `2` into `1`, since all rows with `related` value of `0` and `2` have
  `0` value in the other categories, while the rows with `related` value of `1`
  vary. Credit to Dmitry L in this Udacity Knowledge question: [https://knowledge.udacity.com/questions/64417](https://knowledge.udacity.com/questions/64417).
* On the model  
  The first model created without utilizing `GridSearchCV()` for hyperparameters
  searching  achieved 94.79% accuracy:
  ```shell
  ...
  ...
  ...
  [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
  [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.9s finished
  Accuracy: 0.9479458004915671
  Best Parameters: {}
  -----------------------------------------------------
  Category: related
                precision    recall  f1-score   support

             0       0.69      0.45      0.55      1268
             1       0.84      0.94      0.89      3976

      accuracy                           0.82      5244
     macro avg       0.77      0.69      0.72      5244
  weighted avg       0.81      0.82      0.80      5244

  -----------------------------------------------------
  Category: request
                precision    recall  f1-score   support

             0       0.91      0.98      0.94      4414
             1       0.84      0.48      0.61       830

      accuracy                           0.90      5244
     macro avg       0.88      0.73      0.78      5244
  weighted avg       0.90      0.90      0.89      5244

  -----------------------------------------------------
  ...
  ...
  ...
  ```
  The result on the second model with GridSearchCV() utilized to search for the
  best `clf__estimator__n_estimators` only:
  ```shell
  ...
  ...
  ...
  [Parallel(n_jobs=1)]: Done 200 out of 200 | elapsed:    2.0s finished
  [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
  [Parallel(n_jobs=1)]: Done 200 out of 200 | elapsed:    2.0s finished
  -----------------------------------------------------
  Accuracy: 0.9489045681837444
  Best Parameters: {'clf__estimator__n_estimators': 200}
  -----------------------------------------------------
  Category: related
                precision    recall  f1-score   support
  
             0       0.68      0.46      0.55      1274
             1       0.84      0.93      0.88      3970
  
      accuracy                           0.82      5244
     macro avg       0.76      0.69      0.72      5244
  weighted avg       0.80      0.82      0.80      5244
  
  -----------------------------------------------------
  Category: request
                precision    recall  f1-score   support
  
             0       0.90      0.98      0.94      4358
             1       0.83      0.49      0.62       886
  
      accuracy                           0.90      5244
     macro avg       0.87      0.74      0.78      5244
  weighted avg       0.89      0.90      0.89      5244
  
  -----------------------------------------------------
  ...
  ...
  ...
  ```
  The model achieved accuracy of 94.89%, better than the one with the default
  hyperparameters, with the beset parameter of `clf__estimator__n_estimators==200`.
  
  Both models trained are not included in this repo due to the too big of a file
  size. You need to run the run the ML pipeline as instructed above to obtain it
  and utilize it in the web app.

## File Description ##
* The ETL pipeline and the database are under the `data` directory.
* The machine learning pipeline and the model are under the `models` directory.
* The web application is under the `app` directory.

## Contribution ##
If you want to contribute to this project and make it beter, your help is very
welcome.

The following is the general guide on how to contribute to this project:
   1. Fork this project & clone it on your local machine
   2. Create an *upstream* remote and sync your local copy before you branch
   3. Branch for each piece of work
   4. Do the work
   5. Push to your origin repository
   6. Create a new pull request on GitHub
   
## Acknowledgements ##
A gratitude to [Appen](https://appen.com/) for providing the dataset of this project and
[Udacity](https://www.udacity.com/) for the education and support in providing the
template of the web app and meta-code for the pipelines in this project.

## License ##
The content of thic package is covered under the [MIT License](./license.txt).
