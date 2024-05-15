import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def train_diabetes_model():

    data = pd.read_csv('diabetes.csv')

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model4= KNeighborsClassifier(n_neighbors=5)
    model4.fit(X_train, y_train)

    with open("model4.pkl", "wb") as f:
        pickle.dump(model4, f)
    return X_test, y_test, model4
def plot_roc_curve(X_test, y_test, model):
    # Predict probabilities for the positive class (Outcome = 1) from the test set
    y_probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)

    # Calculate the area under the ROC curve (AUC)
    auc = roc_auc_score(y_test, y_probabilities)

    # Plot the ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.show()
X_test, y_test, model4 = train_diabetes_model()
plot_roc_curve(X_test, y_test, model4)
