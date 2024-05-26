exec(open("/Users/knithin/Desktop/Capstone_2020_23150_2/lgb_classifier.py").read())
import pickle

with open('LightGBM_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="plasma", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()