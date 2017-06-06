import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

#input file containing column indicating presence/absence of y attribute
df = pd.read_csv('ethnic_phoenix_restaurants_data.csv')

#y attribute to predict
y_col = 'ethnic'

feats = df.columns.tolist()
feats.remove('business_id')
feats.remove(y_col)
feats.remove('latitude')
feats.remove('longitude')
                                          
X_train = df[feats]
Y_train = df[y_col]

print 'no. of +ve instances in training set: ', len(df[df[y_col]==1])

#only need this row for predicting on same dataset that model is trained on
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y)

#CODE USED FOR CROSS-VALIDATION IS COMMENTED OUT FOR FINAL PREDICTION

## cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5] }
## ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 
##              'objective': 'binary:logistic'}

## cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
## ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
##              'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 5}   

## model = xgb.XGBClassifier(**ind_params)

## optimized_GBM = GridSearchCV(model, 
##                             cv_params, 
##                              scoring = 'neg_log_loss', cv = 5) 

## optimized_GBM.fit(X_train, Y_train)

## print optimized_GBM.grid_scores_


xgdmat = xgb.DMatrix(X_train, Y_train)

our_params = {'eta': 0.01, 'seed':30, 'subsample': 0.8, 'colsample_bytree': 0.8, 
              'objective': 'binary:logistic', 'max_depth':5, 'min_child_weight':5}
    

## cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
##                 metrics = ['logloss'], # Make sure you enter metrics inside a list or you may encounter issues!
##                 early_stopping_rounds = 100) # Look for early stopping that minimizes error

## print cv_xgb.tail(5)

#num_boost_round determined by output from cv_xgb.tail above
final_gb = xgb.train(our_params, xgdmat, num_boost_round = 565)

importances = final_gb.get_fscore()
print importances

#load test city data
test_df = pd.read_csv('ethnic_bars_stuttgart_restaurants_data.csv')
print 'length of test set: ', len(test_df)

X_val = test_df[feats]
Y_val = test_df[y_col]

#this checks that there are enough restaurants in the y-category to make the prediction stats meaningful.
print 'no. of +ve instances in test set:', len(test_df[test_df[y_col]==1])
                
testdmat = xgb.DMatrix(X_val)
Y_test = final_gb.predict(testdmat)

print log_loss(Y_val, Y_test)
