
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, auc, accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import VotingClassifier
get_ipython().magic('matplotlib inline')


# In[2]:

gbdt_params = {
    'boosting_type': 'gbdt',
    'max_depth' : -1,
    'objective': 'binary',
    'num_leaves': 64,
    'learning_rate': 0.07,
    'n_estimators': 150,
    'num_threads': 4,
    'feature_fraction': 0.8, 
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'min_split_gain': 0.5,
    'is_unbalance': True,
    'bagging_freq': 0,
    'bagging_fraction': 1
}


# In[3]:

rf1_params = {
    'boosting_type': 'rf',
    'max_depth' : 4,
    'objective': 'binary',
    'num_leaves': 16,
    'learning_rate': 0.1,
    'n_estimators': 50,
    'num_threads': 4,
    'feature_fraction': 0.8, 
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'min_split_gain': 1,
    'is_unbalance': True,
    'bagging_freq': 5,
    'bagging_fraction': 0.9
}


# In[4]:

rf2_params = {
    'boosting_type': 'rf',
    'max_depth' : -1,
    'objective': 'binary',
    'num_leaves': 32,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'num_threads': 4,
    'feature_fraction': 0.9, 
    'colsample_bytree': 0.9,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'min_split_gain': 0.5,
    'is_unbalance': True,
    'bagging_freq': 3,
    'bagging_fraction': 0.95
}


# In[5]:

rf3_params = {
    'boosting_type': 'rf',
    'max_depth' : -1,
    'objective': 'binary',
    'num_leaves': 100,
    'learning_rate': 0.1,
    'n_estimators': 150,
    'num_threads': 4,
    'feature_fraction': 0.9, 
    'colsample_bytree': 0.9,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'min_split_gain': 0.1,
    'is_unbalance': True,
    'bagging_freq': 3,
    'bagging_fraction': 0.95
}


# In[6]:

dart_params = {
    'boosting_type':'dart',
    'n_estimators': 150,
    'learning_rate':0.1,
    'colsample_bytree':0.9,
    'subsample':0.9,
    'drop_rate': 0.2,
    'max_drop': 40,
    'max_depth':-1,
    'reg_alpha':0.1,
    'reg_lambda':0.1,
    'is_unbalance': True,
    'bagging_freq': 0,
    'bagging_fraction': 1
}


# In[7]:

xgbst_params = {
    'silent': 1,
    'eta': 0.2,
    'gamma': 0.5,
    'max_depth': 5,
    'max_delta_step': 5,
    'subsample': 0.7,
    'lambda': 1,
    'alpha': 1,
    'scale_pos_weight': 0.01,
    'objective': 'binary:logistic'
}


# In[17]:

cat_params = {
    'learning_rate': 0.1,
    'depth': 8,
    'loss_function': 'Logloss',
    'class_weights': [1, 60],
    'eval_metric': 'F1',
    'iterations': 100,
    'random_seed': 33,
    'l2_leaf_reg': 0.1,
#     'one_hot_max_size': 10,
}


# In[11]:

def preprocess(data, droprow, thres):
    # drop row with nan values 
    if droprow == True:
        drop_columns = ['id', 'date', 'f5', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41',
                        'f42','f43','f44','f45','f46','f47']
        data = data.drop(drop_columns, axis=1)
        data = data.dropna(thresh=thres)
        
    #   data['rate'] = data['f83'] / (data['f84'] + 1)
    #   for col in data.columns:
    #       if col != 'label' and (len(data[col].unique())) < 100:
    #           data[col] = pd.qcut(data[col], 2, duplicates='drop')  
    #           data[col] = pd.factorize(data[col])[0]
    #           onehot = pd.get_dummies(data[col], prefix=col)
    #           data.drop(col, axis = 1, inplace=True)
    #           data = pd.concat([data, onehot], axis=1)
    else:
        drop_columns = ['id', 'date', 'f5', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41',
                        'f42','f43','f44','f45','f46','f47']
        data = data.drop(drop_columns, axis=1)
#         data['rate'] = data['f83'] / (data['f84'] + 1)
        
    return data


# In[12]:

def calculate_score(fpr, tpr):
    
    tpr1_index = np.where(fpr > 0.001)[0][0]
    tpr2_index = np.where(fpr > 0.005)[0][0]
    tpr3_index = np.where(fpr > 0.01)[0][0]
    
    score = tpr[tpr1_index] * 0.4 + tpr[tpr2_index] * 0.3 + tpr[tpr3_index] * 0.3
    
    return score


# In[8]:

train_set = pd.read_csv('atec_anti_fraud_train.csv')


# In[13]:

# seperate labeled and unlabeled data
ts_labeled = train_set[train_set['label'] != -1]
ts_unlabeled = train_set[train_set['label'] == -1]


# In[14]:

# preprocess labeled and unlabeled data
ts_labeled_p = preprocess(ts_labeled, True, 50)
ts_unlabeled_p = preprocess(ts_unlabeled, False, 0)


# In[15]:

# prepare training data
y_labeled = ts_labeled_p['label'].values
X_labeled = ts_labeled_p.drop(['label'], axis=1).values
X_unlabeled = ts_unlabeled_p.drop(['label'], axis=1).values


# In[19]:

# set model parameters
gbdt = lgb.LGBMClassifier(**gbdt_params)
rf1 = lgb.LGBMClassifier(**rf1_params)
rf2 = lgb.LGBMClassifier(**rf2_params)
rf3 = lgb.LGBMClassifier(**rf3_params)
dart = lgb.LGBMClassifier(**dart_params)
xgbst = xgb.XGBClassifier(**xgbst_params)
catt = cat.CatBoostClassifier(**cat_params)


# In[20]:

# set voting classifier
clf = VotingClassifier(estimators=[('gbdt', gbdt), ('rf1', rf1), ('rf2', rf2), ('rf3', rf3),
                                   ('dart', dart), ('xgbst', xgbst), ('catt', catt)], voting='soft')


# In[22]:

# fit model using labeled data
clf_labeled = clf.fit(X_labeled[:1000], y_labeled[:1000])


# In[24]:

# predict unlabeled data using model fitted with labeled data
pred_unlabeled = clf.predict_proba(X_unlabeled)[:, 1]


# In[27]:

# set threshold to get more than 50% positive class in unlabeled data
y_unlabeled_pred = np.where(pred_unlabeled > 0.5, 1, 0)


# In[28]:

# set labels in unlabeled data to predicted values
ts_unlabeled_p['label'] = y_unlabeled_pred


# In[30]:

# concat labeled data and predicted unlabeled data
new_data = pd.concat([ts_labeled_p, ts_unlabeled_p], axis=0)


# In[31]:

# set new training data
new_y = new_data['label'].values
new_X = new_data.drop(['label'], axis=1).values


# In[32]:

# fit model to new data
new_clf = clf.fit(new_X[:1000], new_y[:1000])


# In[33]:

# predict on test data using model fitted with predicted unlabeled data
submission = pd.read_csv('atec_anti_fraud_test_a.csv')
ids = submission['id'].values
sub_x = preprocess(submission, False, 0)
sub_y = new_clf.predict_proba(sub_x.values)[:,1]
output = pd.DataFrame({'id': ids, 'score': sub_y})
output.to_csv("submission.csv", index=False)

