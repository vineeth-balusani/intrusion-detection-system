import pickle
import pandas as pd

dt_pca_pipeline = pickle.load(open("dt_pca_pipeline.pkl", "rb"))
dt_rfe_pipeline = pickle.load(open("dt_rfe_pipeline.pkl", "rb"))

gnb_pca_pipeline = pickle.load(open("gnb_pca_pipeline.pkl", "rb"))
gnb_sfs_pipeline = pickle.load(open("gnb_sfs_pipeline.pkl", "rb"))

logreg_pca_pipeline = pickle.load(open("logreg_pca_pipeline.pkl", "rb"))
logreg_rfe_pipeline = pickle.load(open("logreg_rfe_pipeline.pkl", "rb"))

knn_pca_pipeline = pickle.load(open("knn_pca_pipeline.pkl", "rb"))
knn_rfe_pipeline = pickle.load(open("knn_rfe_pipeline.pkl", "rb"))

randomforest_pca_stackingclassifier = pickle.load(open("randomforest_pca_stackingclassifier.pkl", "rb"))
randomforest_stackingclassifier = pickle.load(open("randomforest_stackingclassifier.pkl", "rb"))

adaboost_pca_stackingclassifier = pickle.load(open("adaboost_pca_stackingclassifier.pkl", "rb"))
adaboost_stackingclassifier = pickle.load(open("adaboost_stackingclassifier.pkl", "rb"))

gbdt_pca_stackingclassifier = pickle.load(open("gbdt_pca_stackingclassifier.pkl", "rb"))
gbdt_stackingclassifier = pickle.load(open("gbdt_stackingclassifier.pkl", "rb"))

xgboost_pca_stackingclassifier = pickle.load(open("xgboost_pca_stackingclassifier.pkl", "rb"))
xgboost_stackingclassifier = pickle.load(open("xgboost_stackingclassifier.pkl", "rb"))


def utility_logical_or(col_a, col_b):
    output = []
    for a, b in zip(col_a, col_b):
        if (a + b >= 1):
            output.append(1)
        else:
            output.append(0)
    return output


def utility_logical_and(col_a, col_b):
    output = []
    for a, b in zip(col_a, col_b):
        if (a==1 and b==1):
            output.append(1)
        else:
            output.append(0)
    return output


def predict_mydata(predict_data, threshold):
    final_results = pd.DataFrame()
    dt_pca_predictions = dt_pca_pipeline.predict(predict_data)
    #PCA Predictions - Logical OR
    test_predictions_ind_pca_or_df = pd.DataFrame()
    test_predictions_ind_pca_or_df["Naive Bayes - Base Learner"] = gnb_pca_pipeline.predict(predict_data)
    test_predictions_ind_pca_or_df["Decision Tree - Base Learner"] = dt_pca_pipeline.predict(predict_data)
    test_predictions_ind_pca_or_df["Logistic Regression - Base Learner"] = logreg_pca_pipeline.predict(predict_data)
    test_predictions_ind_pca_or_df["K Nearest Neighbors - Base Learner"] = knn_pca_pipeline.predict(predict_data)
    test_predictions_ind_pca_or_df["Random Forest - Stacking Classifier"] = randomforest_pca_stackingclassifier.predict(predict_data)
    test_predictions_ind_pca_or_df["AdaBoost - Stacking Classifier"] = adaboost_pca_stackingclassifier.predict(predict_data)
    test_predictions_ind_pca_or_df["XGBoost - Stacking Classifier"] = xgboost_pca_stackingclassifier.predict(predict_data)
    test_predictions_ind_pca_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"] = gbdt_pca_stackingclassifier.predict(predict_data)
    test_predictions_ind_pca_or_df["RF OR AdaBoost"] = utility_logical_or(
        test_predictions_ind_pca_or_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_pca_or_df["AdaBoost - Stacking Classifier"].values)
    test_predictions_ind_pca_or_df["RF OR XGBoost"] = utility_logical_or(
        test_predictions_ind_pca_or_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_pca_or_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_pca_or_df["RF OR GBDT"] = utility_logical_or(
        test_predictions_ind_pca_or_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_pca_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_pca_or_df["AdaBoost OR XGBoost"] = utility_logical_or(
        test_predictions_ind_pca_or_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_pca_or_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_pca_or_df["AdaBoost OR GBDT"] = utility_logical_or(
        test_predictions_ind_pca_or_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_pca_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_pca_or_df["XGBoost OR GBDT"] = utility_logical_or(
        test_predictions_ind_pca_or_df["XGBoost - Stacking Classifier"].values,
        test_predictions_ind_pca_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_pca_or_df["sum_predictions"] = test_predictions_ind_pca_or_df.sum(axis=1)
    test_predictions_ind_pca_or_df["final_prediction"] = test_predictions_ind_pca_or_df["sum_predictions"].apply(
        lambda x: 1 if (x >= threshold) else 0)
    final_results["PCA-Logical OR - Threshold = {}".format(threshold)] = test_predictions_ind_pca_or_df["final_prediction"]
    #PCA Predictions - Logical AND
    test_predictions_ind_pca_and_df = pd.DataFrame()
    test_predictions_ind_pca_and_df["Naive Bayes - Base Learner"] = test_predictions_ind_pca_or_df["Naive Bayes - Base Learner"]
    test_predictions_ind_pca_and_df["Decision Tree - Base Learner"] = test_predictions_ind_pca_or_df["Decision Tree - Base Learner"]
    test_predictions_ind_pca_and_df["Logistic Regression - Base Learner"] = test_predictions_ind_pca_or_df["Logistic Regression - Base Learner"]
    test_predictions_ind_pca_and_df["K Nearest Neighbors - Base Learner"] = test_predictions_ind_pca_or_df["K Nearest Neighbors - Base Learner"]
    test_predictions_ind_pca_and_df["Random Forest - Stacking Classifier"] = test_predictions_ind_pca_or_df["Random Forest - Stacking Classifier"]
    test_predictions_ind_pca_and_df["AdaBoost - Stacking Classifier"] = test_predictions_ind_pca_or_df["AdaBoost - Stacking Classifier"]
    test_predictions_ind_pca_and_df["XGBoost - Stacking Classifier"] = test_predictions_ind_pca_or_df["XGBoost - Stacking Classifier"]
    test_predictions_ind_pca_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"] = test_predictions_ind_pca_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"]
    test_predictions_ind_pca_and_df["RF AND AdaBoost"] = utility_logical_and(
        test_predictions_ind_pca_and_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_pca_and_df["AdaBoost - Stacking Classifier"].values)
    test_predictions_ind_pca_and_df["RF AND XGBoost"] = utility_logical_and(
        test_predictions_ind_pca_and_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_pca_and_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_pca_and_df["RF AND GBDT"] = utility_logical_and(
        test_predictions_ind_pca_and_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_pca_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_pca_and_df["AdaBoost AND XGBoost"] = utility_logical_and(
        test_predictions_ind_pca_and_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_pca_and_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_pca_and_df["AdaBoost AND GBDT"] = utility_logical_and(
        test_predictions_ind_pca_and_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_pca_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_pca_and_df["XGBoost AND GBDT"] = utility_logical_and(
        test_predictions_ind_pca_and_df["XGBoost - Stacking Classifier"].values,
        test_predictions_ind_pca_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)

    test_predictions_ind_pca_and_df["sum_predictions"] = test_predictions_ind_pca_and_df.sum(axis=1)
    test_predictions_ind_pca_and_df["final_prediction"] = test_predictions_ind_pca_and_df["sum_predictions"].apply(
        lambda x: 1 if (x >= threshold) else 0)
    final_results["PCA-Logical AND - Threshold = {}".format(threshold)] = test_predictions_ind_pca_and_df["final_prediction"]
    #Wrapper Predictions - Logical OR
    test_predictions_ind_wrapper_or_df = pd.DataFrame()
    test_predictions_ind_wrapper_or_df["Naive Bayes - Base Learner"] = gnb_sfs_pipeline.predict(predict_data)
    test_predictions_ind_wrapper_or_df["Decision Tree - Base Learner"] = dt_rfe_pipeline.predict(predict_data)
    test_predictions_ind_wrapper_or_df["Logistic Regression - Base Learner"] = logreg_rfe_pipeline.predict(predict_data)
    test_predictions_ind_wrapper_or_df["K Nearest Neighbors - Base Learner"] = knn_rfe_pipeline.predict(predict_data)
    test_predictions_ind_wrapper_or_df["Random Forest - Stacking Classifier"] = randomforest_stackingclassifier.predict(predict_data)
    test_predictions_ind_wrapper_or_df["AdaBoost - Stacking Classifier"] = adaboost_stackingclassifier.predict(predict_data)
    test_predictions_ind_wrapper_or_df["XGBoost - Stacking Classifier"] = xgboost_stackingclassifier.predict(predict_data)
    test_predictions_ind_wrapper_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"] = gbdt_stackingclassifier.predict(predict_data)
    test_predictions_ind_wrapper_or_df["RF OR AdaBoost"] = utility_logical_or(
        test_predictions_ind_wrapper_or_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_wrapper_or_df["AdaBoost - Stacking Classifier"].values)
    test_predictions_ind_wrapper_or_df["RF OR XGBoost"] = utility_logical_or(
        test_predictions_ind_wrapper_or_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_wrapper_or_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_wrapper_or_df["RF OR GBDT"] = utility_logical_or(
        test_predictions_ind_wrapper_or_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_wrapper_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_wrapper_or_df["AdaBoost OR XGBoost"] = utility_logical_or(
        test_predictions_ind_wrapper_or_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_wrapper_or_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_wrapper_or_df["AdaBoost OR GBDT"] = utility_logical_or(
        test_predictions_ind_wrapper_or_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_wrapper_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_wrapper_or_df["XGBoost OR GBDT"] = utility_logical_or(
        test_predictions_ind_wrapper_or_df["XGBoost - Stacking Classifier"].values,
        test_predictions_ind_wrapper_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_wrapper_or_df["sum_predictions"] = test_predictions_ind_wrapper_or_df.sum(axis=1)
    test_predictions_ind_wrapper_or_df["final_prediction"] = test_predictions_ind_wrapper_or_df[
        "sum_predictions"].apply(lambda x: 1 if (x >= threshold) else 0)
    final_results["Wrapper-Logical OR - Threshold = {}".format(threshold)] = test_predictions_ind_wrapper_or_df["final_prediction"]
    #Wrapper Predictions - Logical AND
    test_predictions_ind_wrapper_and_df = pd.DataFrame()

    test_predictions_ind_wrapper_and_df["Naive Bayes - Base Learner"] = test_predictions_ind_wrapper_or_df[
        "Naive Bayes - Base Learner"]
    test_predictions_ind_wrapper_and_df["Decision Tree - Base Learner"] = test_predictions_ind_wrapper_or_df[
        "Decision Tree - Base Learner"]
    test_predictions_ind_wrapper_and_df["Logistic Regression - Base Learner"] = test_predictions_ind_wrapper_or_df[
        "Logistic Regression - Base Learner"]
    test_predictions_ind_wrapper_and_df["K Nearest Neighbors - Base Learner"] = test_predictions_ind_wrapper_or_df[
        "K Nearest Neighbors - Base Learner"]
    test_predictions_ind_wrapper_and_df["Random Forest - Stacking Classifier"] = test_predictions_ind_wrapper_or_df[
        "Random Forest - Stacking Classifier"]
    test_predictions_ind_wrapper_and_df["AdaBoost - Stacking Classifier"] = test_predictions_ind_wrapper_or_df[
        "AdaBoost - Stacking Classifier"]
    test_predictions_ind_wrapper_and_df["XGBoost - Stacking Classifier"] = test_predictions_ind_wrapper_or_df[
        "XGBoost - Stacking Classifier"]
    test_predictions_ind_wrapper_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"] = \
    test_predictions_ind_wrapper_or_df["Gradient-Boosted Decision Tree - Stacking Classifier"]
    test_predictions_ind_wrapper_and_df["RF AND AdaBoost"] = utility_logical_and(
        test_predictions_ind_wrapper_and_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_wrapper_and_df["AdaBoost - Stacking Classifier"].values)
    test_predictions_ind_wrapper_and_df["RF AND XGBoost"] = utility_logical_and(
        test_predictions_ind_wrapper_and_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_wrapper_and_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_wrapper_and_df["RF AND GBDT"] = utility_logical_and(
        test_predictions_ind_wrapper_and_df["Random Forest - Stacking Classifier"].values,
        test_predictions_ind_wrapper_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_wrapper_and_df["AdaBoost AND XGBoost"] = utility_logical_and(
        test_predictions_ind_wrapper_and_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_wrapper_and_df["XGBoost - Stacking Classifier"].values)
    test_predictions_ind_wrapper_and_df["AdaBoost AND GBDT"] = utility_logical_and(
        test_predictions_ind_wrapper_and_df["AdaBoost - Stacking Classifier"].values,
        test_predictions_ind_wrapper_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_wrapper_and_df["XGBoost AND GBDT"] = utility_logical_and(
        test_predictions_ind_wrapper_and_df["XGBoost - Stacking Classifier"].values,
        test_predictions_ind_wrapper_and_df["Gradient-Boosted Decision Tree - Stacking Classifier"].values)
    test_predictions_ind_wrapper_and_df["sum_predictions"] = test_predictions_ind_wrapper_and_df.sum(axis=1)
    test_predictions_ind_wrapper_and_df["final_prediction"] = test_predictions_ind_wrapper_and_df[
        "sum_predictions"].apply(lambda x: 1 if (x >= threshold) else 0)
    final_results["Wrapper-Logical AND - Threshold = {}".format(threshold)] = test_predictions_ind_wrapper_and_df["final_prediction"]
    for feature in ["PCA-Logical OR", "PCA-Logical AND", "Wrapper-Logical OR", "Wrapper-Logical AND"]:
        feature = feature + " - Threshold = {}".format(threshold)
        final_results[feature] = final_results[feature].apply(lambda x:"normal" if x==0 else "attack")
    return final_results




