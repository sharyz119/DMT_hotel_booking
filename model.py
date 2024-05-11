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
    


#Definition of LambdaMART Model 
def lambda_mart(Train_features, Train_scores, Train_qids, Val_features, Val_scores, Val_qids, stop, num_estim):
   
    metric = pyltr.metrics.NDCG(k=5)
    monitor = pyltr.models.monitors.ValidationMonitor(Val_features, Val_scores, Val_qids, metric=metric,
                                                      stop_after=stop)
    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=num_estim,
        max_features=0.5,
        learning_rate=0.02,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
    )
    #Fit Model
    model.fit(Train_features, Train_scores, Train_qids, monitor=monitor)
    return model


#Store Output
def store_output(Epred_r, new_test_set):
    predictions_df = pd.DataFrame(Epred_r)
    submiss_test_set = pd.DataFrame(new_test_set["srch_id"])
    submiss_test_set.columns = ["srch_id"]
    submiss_test_set["ranking"] = predictions_df
    submiss_test_set["prop_id"] = new_test_set["prop_id"]
    test_set_submission_result = submiss_test_set.groupby(["srch_id"]).apply(
        lambda x: x.sort_values(["ranking"], ascending=False)).reset_index(drop=True)
    test_set_submission_result = test_set_submission_result.drop("ranking", axis=1)
    test_set_submission_result.to_csv("RESULT_to_submit.csv", index=False)



def main():
    print("Open the files...")
    #Import the train and validation set in the SVMlight format
    full_train = open("C://Users//eva//Documents//AI//Data_mining//Data_mining_assignment_2//Full_train_lm_split.txt")
    full_valid = open("C://Users//eva//Documents//AI//Data_mining//Data_mining_assignment_2//Full_validation_lm_split.txt")
    full_test = open("C://Users//eva//Documents//AI//Data_mining//Data_mining_assignment_2//Preprocessed_test_set.txt")

    #Load test set in normal format
    print("Load the normal test set...")
    path = "C://Users//eva//Documents//AI//Data_mining//Data_mining_assignment_2//"
    new_test_set = pd.read_csv(path + "/New_test_set_full.csv")
    new_test_set = new_test_set.drop(["Unnamed: 0"], axis=1)

    #Split in scores, feature and search_ids
    print("Load the SVMlight train set...")
    Train_features, Train_scores, Train_qids, _ = pyltr.data.letor.read_dataset(full_train)
    print("Load the SVMlight validation set...")
    Val_features, Val_scores, Val_qids, _ = pyltr.data.letor.read_dataset(full_valid)
    print("Load the SVMlight test set...")
    Test_features_r, Test_scores_r, Test_qids_r, _ = pyltr.data.letor.read_dataset(full_test)

    full_test.close()
    full_train.close()
    full_valid.close()


    #Parameters of LambdaMART
    stop = 100  #after how many equal score (no imporvement) stop
    num_estimators = 2000  #number of trees to use. HIGHER it is LONGER IT takes to run the script

    print("\nStart training of LambdaMART...\n")
    trained_model = lambda_mart(Train_features, Train_scores, Train_qids, Val_features, Val_scores, Val_qids, stop,
                                num_estimators)
    
    #Predict scores for ranking
    print("\nMaking the predictions...\n")
    Epred_test = trained_model.predict(Test_features_r)
   
    #Store the prediction to file
    store_output(Epred_test, new_test_set)
    print("Feature importances: \n")
    print(trained_model.feature_importances_)
    print("Estimators fitted: \n")
    print(trained_model.estimators_fitted_)


if __name__ == "__main__":
    main()



