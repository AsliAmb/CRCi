
import os
import joblib
import pandas as pd
import numpy as np

"""
===================================
FUNCTIONS
===================================
"""



def validate(data):
    """
    Directs user 
    input:
        data: PSI values (samples in rows, observations (PSI values) in columns) 
    output:
        A warning is printed about analysis mode selection
    """
    if data.shape[0] >= 15:
        return "You have 15 or more samples, we recommend that you use multi-sample CMS predictor"
    else:
        return "You have less than 15 samples, we recommend that you use single-sample CMS predictor"
        
   

def min_max_scaling(column):
    """ Min-max scaling of input data with samples < 15"""
    return (column - column.min()) / (column.max() - column.min())

def softmax(x):
    """ Calculate probabilities for each set of scores(logits)"""
    e_x = np.exp(x)
    if x.ndim == 1:
        return e_x / e_x.sum()
    elif x.ndim == 2:
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    else:
        raise ValueError("Input must be a 1D or 2D array")
        
        
def predict_probabilities(coefs, intercepts, X):
    """Predict class probabilities for new values X using the model coefficients and intercepts. """
    scores = np.dot(X, coefs.T) + intercepts
    probabilities = softmax(scores)
    return probabilities


def predict_CMS(data,mode='single'):
    """
    Given a analysis mode and input data containing PSI values 29 CMS predictive 
    input:
        data: PSI values (samples in rows, observations (PSI values) in columns). The PSI values for the following
        Exon-skipping events should be included. The numbers represent start and end coordinates for the inclusion 
        exon in genome assembly GRCh38. 

       'AURKA_56388686_56388784', 'ITGAE_3723287_3723383',
       'TPM1_63061197_63061273', 'PXN_120224642_120224727',
       'C16orf13_635517_635774', 'MACROD1_63998837_63998872',
       'SEC31A_82842184_82842481', 'SLMAP_57925844_57925934',
       'MYOF_93392916_93392955', 'MRRF_122285779_122285946',
       'EPB41L3_5394676_5394793', 'KALRN_124637207_124637303',
       'LUC7L_228332_228402', 'MDM4_204537458_204537497',
       'ENAH_225504990_225505053', 'USPL1_30621072_30621239',
       'XPO1_61525269_61525333', 'CCDC112_115269702_115269798',
       'FNIP1_131710577_131710661', 'PBX1_164820071_164820184',
       'KIF13A_17771113_17771218', 'MPRIP_17180606_17180669',
       'MYO6_75898372_75898411', 'FRYL_48593929_48594016',
       'CEP78_78265858_78265906', 'FNBP1_129923843_129923996',
       'KIAA1217_24542692_24542770', 'ANKRD26_27044156_27044190',
       'SORBS1_95414493_95414862'
       
       mode: Analysis mode. For samples > 14, multi-sample model is used for CMS prediction. Otherwise,
       single-sample mode is used {"multi","sample"}
       
    output:
        probabilities: probabilities for each CMS is returned. The class membership is decided by the maximum
        probability
       
 

"""

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if mode != "multi" and mode != "single":
        print ("Please choose a mode. If n > 14, analysis mode should be `multi`, else the mode should be `single`")
    else:
        print(validate(data))
        if mode == "multi":
            if data.shape[0] < 5:
                 raise ValueError("Please use the single analysis mode")
                 

            else:

                model_path = os.path.join(dir_path, 'models/cohort.joblib')
                model = joblib.load(model_path)
                data = data[data.columns[data.columns.isin(model.feature_names_in_)]]
                data = data[model.feature_names_in_]
                data = (data - data.mean()) / data.std()
                ind = data.index
        elif mode =="single":
            model_path = os.path.join(dir_path, 'models/single_sample.joblib')
            model = joblib.load(model_path)
            data = data[data.columns[data.columns.isin(model.feature_names_in_)]]
            data = data[model.feature_names_in_]
            if data.shape[0] > 1 :
                ind = data.index
                data = data.apply(min_max_scaling, axis =1)
            else:
                ind = data.index
                data = min_max_scaling(data.iloc[0].values)

        else:
            print ("Please enter a valid analysis mode")
        prbs = predict_probabilities (model.coef_, model.intercept_, data)
        if prbs.ndim > 1 :
            cms_class = np.argmax(prbs, axis=1)
            pred_CMS = ["CMS"+str(i+1) for i in cms_class]
            probabilities = pd.DataFrame(prbs)
        else:
             cms_class = np.argmax(prbs)
             pred_CMS = "CMS" + str(cms_class + 1)
             probabilities = pd.DataFrame([prbs])
        
    
        probabilities.columns = ["pCMS1","pCMS2","pCMS3","pCMS4"]
        probabilities['predicted.CMS'] = pred_CMS
        probabilities.index = ind
        return(probabilities)    
    
    
