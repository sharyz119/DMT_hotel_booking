#Import Libraries 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


#Definition of different models 
def model():
    #Model 1: Random Forest 
    # return RandomForestClassifier(n_estimators=50, 
                                            # verbose=2,
                                            # n_jobs=1,
                                            # min_samples_split=10,
                                            # random_state=1)
    
    
    #Model 2: SVR 
    # return SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    
    #Model 3: Gradient Boosting                                         
    return GradientBoostingClassifier(loss='deviance', 
                                        learning_rate=0.1, 
                                        n_estimators=100, 
                                        subsample=1.0, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, 
                                        max_depth=3, 
                                        init=None, 
                                        random_state=None, 
                                        max_features=None, 
                                        verbose=0)
    

    

#Function to tune parameters
def tune_parameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)
    return grid_search.best_estimator_


if __name__ == "__main__":
    dataset = pd.read_csv("test_set_VU_DM.csv") #Load the dataset into a DataFrame
    
    # Prepare the data - adjust these lines as necessary to match your dataset structure
    X = dataset.drop('target_column_name', axis=1)  #Replace 'target_column_name' with the name of your target column
    y = dataset['target_column_name']  #Replace 'target_column_name' with the actual target column name


    #Choose which model to tune
    selected_model = model('SVR')  #Can be 'RandomForest', 'SVR', or 'GradientBoosting'

    #Define parameter grid based on selected model
    if isinstance(selected_model, SVR):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    elif isinstance(selected_model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif isinstance(selected_model, GradientBoostingClassifier):
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.5],
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 10]
        }

    #Tune parameters
    best_model = tune_parameters(selected_model, param_grid, X, y)
    
