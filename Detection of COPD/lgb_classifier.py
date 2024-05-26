exec(open("/Users/knithin/Desktop/Capstone_2020_23150_2/xgb_matrix.py").read())
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
params = {
    'objective': 'multiclass',
    'num_class': 6,  
    'num_leaves': 50,  
    'learning_rate': 0.1,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'seed': 42,
    'n_estimators': 1000,  
    'metric': 'multi_error',  
    'verbosity': -1  
}

lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X_train, y_train)


y_pred = lgb_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


print("\nClassification Report:\n", classification_report(y_test, y_pred))
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    'objective': 'multiclass',
    'num_class': 6, 
    'num_leaves': 50,  
    'learning_rate': 0.1,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'seed': 42,
    'n_estimators': 1000,  
    'metric': 'multi_error', 
    'verbosity': -1  
}


lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X_train, y_train)


y_pred = lgb_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


print("\nClassification Report:\n", classification_report(y_test, y_pred))