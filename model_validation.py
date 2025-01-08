import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d' if not normalize else '.2f',
                cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

# Assuming you have your model and data preprocessing steps defined above
# Here's how the validation should be structured:

def validate_model(model, X_val, preprocessor):
    # Preprocess validation data the same way as training data
    X_val_prep = preprocessor.transform(X_val)
    
    # Make predictions
    predictions = model.predict(X_val_prep)
    predictions = [1 if x > 0.5 else 0 for x in predictions]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, predictions)
    print('Val Accuracy = %.2f' % accuracy)
    
    # Create confusion matrix
    confusion_mtx = confusion_matrix(y_val, predictions)
    labels = {0: 'Class 0', 1: 'Class 1'}  # Adjust these labels as needed
    cm = plot_confusion_matrix(confusion_mtx, classes=list(labels.values()), normalize=False)
    plt.show()
    
    return accuracy, confusion_mtx