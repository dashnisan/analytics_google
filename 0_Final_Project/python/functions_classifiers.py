#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dashnisan
"""

################################################################################
#                        Data manipulation
import numpy as np
import pandas as pd
# Regular expressions:
import re 
# Data sets:
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
# data preprocessing:
import sklearn.preprocessing as prepro

################################################################################
#                        Statistics
from scipy import stats

################################################################################
#                         Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from xgboost import plot_importance

################################################################################
#                         Data modeling

# Models:
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Grid Search Cross-Validation:
from sklearn.model_selection import GridSearchCV

################################################################################
#                         Model metrics:
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, PrecisionRecallDisplay
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

################################################################################

# Model export/import:
import pickle
import joblib # more efficient at serializing large numpy arrays                

################################################################################

def string_convert_to_lower(match_obj):
    '''
    Replacement function to convert uppercase letter to lowercase.
    To be used as argument in re.sub(), for example:
      temp_string = re.sub(r'[A-Z]', convert_to_lower, feature)
    '''
    if match_obj.group() is not None:
        return match_obj.group().lower()
        
################################################################################               


def df_boxplot(df):
    ''' plot boxplots for all features in data frame.
    df is the input dataframe for plotting the box plots'''
    sns.set()
    df.boxplot(figsize=(12,5))
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.yscale('log')
    plt.show()
    
########################################################################################
    
def plothist_fill_hue(dataframe, feature, hue):
    ''' Plot histogram with hue fill mode'''
    plt.figure(figsize=(6,3))
    sns.histplot(data=dataframe, y=feature, hue=hue, multiple='fill')
    #sns.histplot(data=dataframe, y=feature, stat='percent', hue='left')
    plt.title('Relative Frequency of {} with hue={} '.format(feature, hue))
    plt.show()
    

########################################################################################

def drop_outliers(df, outliers_columns,
                  quantile_low: float = 0.25,
                  quantile_hi: float = 0.75,
                  iqr_factor: float = 1.5):
    '''
    drops the rows of a data frame with outliers for a given numerical
    colum (either continuous or discrete).
    Arguments:
    * df: data frame to process
    * outliers_columns: a list with the column names of df 
      where the outliers should be looked for.
    The dropping is performed one column after the other.
    The criterion is defined by quantiles given as arguments:
    * quantile_low: low limit of interquartile range iqr
    * quantile_hi: hi limit of interquartile range iqr
    The limits are for dropping are calculated with igr and a
    factor set to 1.5*iqr:
        upper_limit = quantiles.iloc[1] + iqr_factor * iqr
        lower_limit = quantiles.iloc[0] - iqr_factor * iqr
    Returns:
    df: a data frame where the outliers were removed
        
    '''
    for column in outliers_columns:
        quantiles = df[column].quantile([quantile_low, quantile_hi])
        iqr = quantiles.iloc[1] - quantiles.iloc[0]
        upper_limit = quantiles.iloc[1] + iqr_factor * iqr
        lower_limit = quantiles.iloc[0] - iqr_factor * iqr
        if lower_limit < 0:
            lower_limit = 0

        original_max = df[column].max()
        original_min = df[column].min()
        temp_mask = (df[column] <= lower_limit) | (df[column] >= upper_limit)
        df = df.drop(df[temp_mask].index)
        new_max = df[column].max()
        new_min = df[column].min()
        print('Outliers for {}:\n max original = {:.3e}\n max_nooutliers = {:.3e}'
          .format(column, original_max, new_max))
        print(' min original = {:.3e}\n min_nooutliers = {:.3e}'
          .format(original_min, new_min))
        print('Lower limit is: {:1.1e}'.format(lower_limit))
        print('Upper limit is: {:1.1e}'.format(upper_limit))
        print('dropped {} rows from data frame\n'.format(len(temp_mask[temp_mask == True])))
    return df

    
    
########################################################################################

def splitset_train_validation_test(Xin, yin,
                                   validation_size: float = 0.20,
                                   test_size: float = 0.20,
                                   random_state: float = 44):
    '''Splitting with stratification  on y
    The X and y for the model are returned as dictionaries
    with 'train', 'valid' and 'test' as keys
    Input set divided into train0 and test sets according to test_size
    train0 set divided into train and validation according to validation_size
    Set sizes are relative to original set size'''
    
    X=dict()
    y=dict()
    X_train0, X['test'], y_train0, y['test'] = train_test_split(Xin, yin, 
                                                                      test_size=test_size,
                                                                      stratify=yin,
                                                                      random_state=random_state)
    # Calculate factor for validation set from user input:
    b = validation_size / (1 - test_size)
    X['train'], X['valid'], y['train'], y['valid'] = train_test_split(X_train0, y_train0, 
                                                                      test_size = b,
                                                                      stratify = y_train0,
                                                                      random_state=random_state) 
    # check spliting ratios:
    for set in ['train', 'valid', 'test']:
        set_ratio = len(y[set]) / len(yin)
        print('{} set size: {} of original data set'.format(set, set_ratio))
        print('Value distribution in y vector: \n {}\n'.format(y[set].value_counts(normalize=True)))
    return X,y
    print('exiting split')
    
########################################################################################
    
def column_encode(df, column, map_dictionary):
    '''Inserts a new column after the given one
    transformed to the values given in dictionary'''

    new_column = df[column].map(map_dictionary)
    new_colum_name = column +'_encoded'
    df.insert(df.columns.get_loc(column)+1, column=new_colum_name, value=new_column)

    # Check proportions are kept:
    a = df[column].value_counts(normalize=True).values.all()
    b = df[new_colum_name].value_counts(normalize=True).values.all()
    if a == b:
        print('Column transform success with {} and proportions\n {}\n'.format(map_dictionary, df[column].value_counts(normalize=True)))
    else:
        print('Column transform failure with {}\n {}\n and {}\n'.format(map_dictionary, a, b))
        
########################################################################################        

def logistic_regression_feature_coeff_importance(lr_model_object):

    print('Explanation:\ntheta_i provides the specific contribution of feature i to the\nvariation\
    of log(odds) or log(P(Y=1) / P(Y=0)) in the equation:\n\n         \
    log(odds) = theta_vector dot X_vector\n\n\
    The absolute value of theta provides a measure of the feature importance in the model')
    
    coeffs = lr_model_object.coef_[0][:]
    abscoeffs = np.abs(coeffs)
    temp_array = np.transpose([coeffs, abscoeffs])

    indices = list(lr_model_object.feature_names_in_)
    columns=[r'$\theta$', r'$|\theta|$']
    model_coeff = pd.DataFrame(data=temp_array, index=indices, columns=columns)

    sns.set()
    plt.figure(figsize=[8,5])
    order = list(model_coeff.sort_values(by=r'$|\theta|$', ascending=False).index)
    sns.barplot(data=model_coeff,
                y=model_coeff.index,
                x=model_coeff[r'$|\theta|$'],
                order=order)
    plt.xlabel(r'$|\theta|$ Regression Coefficient')
    plt.title(r'Logistic Regression Features Importances in Model $log[\frac{P(y=1)}{P(y=0)}] = \vec \theta * \vec X$')
    plt.show()
                      
    return model_coeff.sort_values(by=r'$|\theta|$', ascending=False)
    
    
########################################################################################


def logistic_regression_probabilities(lr_model_object, X_set):
    ''' calaculate probabilities for an already fitted lr model for the X_set,
    normally the train set'''
     
    classes = lr_model_object.classes_
    columns=list()
    for pclass in classes:
        pstring = 'P_' + str(pclass)
        columns.append(pstring)
    probas =  pd.DataFrame(data=lr_model_object.predict_proba(X_set),
                           columns=columns)
    return probas
    

########################################################################################

def logistic_regression_logits(lr_probabilities, 
                               X_set, features_list, 
                               p1: str = 'P_1',
                               p0: str = 'P_0',
                              ):
                              
    '''Calculate the logits = log(P_1/P_0)
    and plot against each feature. This is a projection
    on the axis of the feature'''
    
    logits = np.log(lr_probabilities[p1]/lr_probabilities[p0])

    # Plot the projection of the calculated logits for model for each x variable:
    for feature in features_list:
        fig, axes =  plt.subplots(1,2, figsize=(8,3))
        sns.lineplot(data=X_set, x=feature, y=logits, errorbar='sd', label='logit mean +- std', ax=axes[0])
        axes[0].set_title('Logit lineplot')
        sns.regplot(data=X_set, x=feature, y=logits, scatter_kws={'s': 2, 'alpha': 0.5}, ax=axes[1])
        axes[1].set_title('Logit regplot')
        plt.tight_layout()
        plt.show()


########################################################################################

def binary_classifier_scores(y_reference, y_prediction):
    
    model_scores = ['f1_score', 'accuracy_score', 'precision_score', 'recall_score']
    temp_scores = list()
    for score in model_scores:
        score_value = globals()[score](y_reference, y_prediction)
        # line above a bit tricky, see:
        # https://stackoverflow.com/questions/4018953/whats-the-way-to-call-a-function-dynamically-in-python
        #print(score_value)
        temp_scores.append(score_value)
    temp_scores = dict(zip(model_scores, temp_scores))
    results = pd.DataFrame(data=temp_scores, columns=model_scores, index=[0])
    #print(results)
    return results

    
########################################################################################

def models_join_scores_tables(modelscores, modelnames):
    ''' Joins dataframes with scores for results from function 
    binary_classifier_scores 
    modelscores: list of the variable names of the dfs with the scores
    modelnames: names as should be displayed as indices of result df.
    '''
    indices = list(modelnames)
    table = pd.DataFrame()
    for modelscore in modelscores:
        table = pd.concat([table, modelscore], axis=0)
    table.index = indices
    return table
    

########################################################################################

def compare_feature_importances(modelimportances, modelnames):

    '''Joins the importances dfs as output by function
    logistic_regression_feature_coeff_importance
    If only two models are supplied, then barplot
    feature importances'''
    
    columns = indices = list(modelnames)

    table = pd.DataFrame()
    for modelimportance in modelimportances:
        table = pd.concat([table, modelimportance], axis=1)
    table.columns = indices

    # plot importances as barplot for twlo models only:
    if len(modelimportances) == 2:
        # Numbers of pairs of bars you want
        N = len(table)

        # Position of bars on x-axis
        ind = np.arange(N)
        # Figure size
        plt.figure(figsize=(10,5))
        # Width of a bar 
        width = 0.3       

        # Plotting
        plt.bar(ind, table.iloc[:,0], width, label=modelnames[0])
        plt.bar(ind + width, table.iloc[:,1], width, label=modelnames[1])

        plt.xlabel('Feature')
        plt.ylabel('Feature importance')
        plt.title('Feature Importances for Selected Models')

        # xticks()
        # First argument - A list of positions at which ticks should be placed
        # Second argument -  A list of labels to place at the given locations
        plt.xticks(ind + width / 2, list(table.index), rotation=45, horizontalalignment='right')

        # Finding the best position for legends and putting it
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    return table

########################################################################################

def cm_2_barplot(cm_array, labels_dict):

    '''Generates barplot form confusion matrix object
    labels_dict is used to map NEGATIVE and POSITIVE to user 
    categories in the usecase, e.g:
    NEGATIVE:ABSENT
    POSITIVE:PRESENT
    '''
    
    
    tn = cm_array[0,0]
    fp = cm_array[0,1]
    fn = cm_array[1,0]
    tp = cm_array[1,1]

    table = pd.DataFrame()
    columns = ['COUNT', 'PREDICTION_CLASS', 'TRUTH_CLASS']
    index=['TN','FP','FN','TP']
    data=[[tn, 'NEGATIVE', 'TRUE'],
          [fp, 'POSITIVE', 'FALSE'],
          [fn, 'NEGATIVE', 'FALSE'],
          [tp, 'POSITIVE', 'TRUE']]
    table = pd.DataFrame(data, columns=columns, index=index)
    table = table.replace(labels_dict)
    #print(table)
    #sns.set()
    plt.figure(figsize=(6,3))
    sns.catplot(data=table,
               col= 'PREDICTION_CLASS',
               y='COUNT',
               x='TRUTH_CLASS',
               kind='bar')
    plt.tight_layout()
    plt.show()
    return table

########################################################################################


def table_model_scores(model_name:str, model_object, model_metrics, order_metric:str):
    '''
    Generate table of test scores for GridSearchCV model
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        order_metric (string): precision, recall, f1, or accuracy. For presenting order

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Check metrics to extract are included in expected scores:
    expected_metrics = set(['accuracy', 'precision', 'recall', 'f1'])
    model_metrics = set(scoring)
    if expected_metrics != model_metrics:
        print('Error: Metrics unmatch:\n expected {}, got {}\n'.format(expected_metrics, model_metrics))
        return 1
    else:
         print('Metrics match, metrics to extract are:\n expected {}\n'.format(expected_metrics, model_metrics))
        

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy',
                   }
    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[order_metric]].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    df_table = pd.DataFrame(
                         {'model': [model_name],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'precision': [precision],
                          'recall': [recall],                          
                          },
                         )

    return df_table


########################################################################################

def cv_best_fit_summary(model_object):
    '''Get the best results of cross validation:
    Arguments:
    model_object: the model.
    Thes model should be under the key ['model'] of model_dict_name'''

    print('Best score: {}\n'.format(model_object.best_score_))
    print('Best estimator: {}\n'.format(model_object.best_estimator_))
    print('Best parameters: {}\n'.format(model_object.best_params_))

    # store results separately:
    cv_results = pd.DataFrame(model_object.cv_results_)
    return cv_results



########################################################################################

def cv_best_classifier_scores_params_ref_param(model_object, ref_metric:str, gridsearchcv_params:dict):
    '''
    Generate table of test scores for GridSearchCV model
    Arguments:
        
        * model_object: a fit GridSearchCV object
        * ref_metric (string): precision, recall, f1, or accuracy. Is the metric according to
        the best estimator is searched (manually, independent of attribute best_estimator_).
        It is also used for presenting order.
        Example: If "f1" is give3n it searches the cv results and looks for the estimator 
        with the highest f1-score in the table.
        * gridsearchcv_params: the dictionary with the hyperparamters names and values given as
        input argument for GridSearchCV

    Returns:
    * best_scores: a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    * best_estimator_export: a dictionary with 2 elements: 
        1. a small data frame with the  index of the best estimator in the cv_results_ data frame.
        2. a data frame with the values of the hyperparameters given in the cv. 
    * cv_results_view: a view of cv_results_ for all the estimators of cv including the 
      metrics f1, accuracy, precision and recall and the cv-parameters. Other parameters
      are not shown. Ordered in descending order according to highest ref_metric.
    '''
        

    # Create dictionary that maps common metrics names to actual metric name in GridSearchCV:
    metric_dict = {
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall', 
                   }
    print('Metrics to be extracted from .cv_results are:\n {}\n'.format(metric_dict.values()))
    # Get all the results from the CV and put them in a df:
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score:
    index_highest_score = cv_results[metric_dict[ref_metric]].idxmax()
    best_estimator_results = cv_results.iloc[index_highest_score, :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of scores as data frame:
    best_scores = pd.DataFrame(
                         {
                          'f1_score': [f1],
                          'accuracy_score': [accuracy],
                          'precision_score': [precision],
                          'recall_score': [recall],                          
                          },
                         )

    # Create separate table with CV parameters of the best estimator according to ref_metric:
    # These parameters are given in the dictionary as input argument for GridSearchCV:

     # parameters without prefix 'param_':
    gscv_params_series = pd.Series(gridsearchcv_params.keys())
    prefix_series = pd.Series(['param_']*5)
    # parameters with prefix 'param' as found in cv_results_:
    gscv_params_series_patched = prefix_series.str.cat(gscv_params_series) 
    best_params = pd.DataFrame(best_estimator_results[gscv_params_series_patched])
    #print(best_params.columns)
    best_params.columns = ['value']
    best_params.index = list(gscv_params_series)

    # reshape df to be able to translate to dict for fitting:
    #best_params.insert(0, 'parameter',list(best_params.index))
    #best_params.index = np.arange(0, len(best_params))


    
    # df with index of best_estimator in cv_results_:
    #colum_name = best_params.columns[0]
    colum_name = 'data_frame_index'
    index_df = pd.DataFrame([{ colum_name : index_highest_score}])
    index_df.index = ['index_best_estimator_cv_results_']
    
    #best_params = pd.concat([temp_df, best_params])

    # Export data frames to dictionary:
    best_estimator_export = dict()
    best_estimator_export ={'index_best_estimator_cv_results_':index_df,
                           'params':best_params.transpose()}

    # view of selected parameters and scores from cv_results_:
    score_names = list(metric_dict.values())
    cv_results_view = cv_results[ score_names + list(gscv_params_series_patched) ]
    cv_results_view = cv_results_view.sort_values(by='mean_test_f1', ascending=False)

    return best_scores, best_estimator_export, cv_results_view


########################################################################################

def table_feat_importance_native(model_object):
    '''
    Returns a table as a data frame with relative importances
    of features for models with attributes:
            .feature_names_in_
            .feature_importances_
    Ordered in descending order
    '''
    
    index = list(model_object.feature_names_in_)
    #print(index)
    table_df = pd.DataFrame(data=model_object.feature_importances_,
                            index=index,
                            columns=['feature_importance'])
    table_df = table_df
    return table_df.sort_values(by='feature_importance', ascending=False)


########################################################################################

