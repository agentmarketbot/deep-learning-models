import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing_utils import prepare_data, predict_with_validation
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d' if not normalize else '.2f',
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt.gcf()

def evaluate_model(model, X_train, X_val, y_val, labels):
    try:
        # Prepare data with consistent preprocessing
        X_train_prep, X_val_prep = prepare_data(X_train, X_val)
        
        # Make predictions with validation
        predictions = predict_with_validation(model, X_val_prep)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, predictions)
        print('Validation Accuracy = %.2f' % accuracy)
        
        # Create confusion matrix
        confusion_mtx = confusion_matrix(y_val, predictions)
        cm = plot_confusion_matrix(
            confusion_mtx,
            classes=list(labels.items()),
            normalize=False,
            title='Confusion Matrix'
        )
        
        return accuracy, cm
        
    except ValueError as ve:
        print(f"Validation Error: {str(ve)}")
        raise
    except RuntimeError as re:
        print(f"Runtime Error: {str(re)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

# Example usage:
"""
# Initialize your model and data
model = YourModel()
X_train = ...
X_val = ...
y_val = ...
labels = {0: 'Class A', 1: 'Class B'}

# Evaluate the model
try:
    accuracy, confusion_matrix_plot = evaluate_model(model, X_train, X_val, y_val, labels)
    plt.show()
except Exception as e:
    print(f"Model evaluation failed: {str(e)}")
"""