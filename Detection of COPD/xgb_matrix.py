exec(open("/Users/knithin/Desktop/Capstone_2020_23150_2/xgb_classifier.py").read())
import pickle

with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="viridis", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
import pickle
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="viridis", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()