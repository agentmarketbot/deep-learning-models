# Solution for Issue #120: NameError: name 'X_val_prep' is not defined

Fixes #120

## Problem
The error occurs because the validation data wasn't preprocessed before being used for prediction. The validation data must go through the same preprocessing steps as the training data.

## Solution

1. **Save your preprocessor during training:**
```python
# During training
from sklearn.preprocessing import StandardScaler  # or whatever preprocessor you're using

# Create and fit preprocessor
preprocessor = StandardScaler()
preprocessor.fit(X_train)

# Preprocess training data
X_train_prep = preprocessor.transform(X_train)

# Train model
model.fit(X_train_prep, y_train)
```

2. **Preprocess validation data:**
```python
# During validation
X_val_prep = preprocessor.transform(X_val)

# Make predictions
predictions = model.predict(X_val_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]

# Calculate metrics
accuracy = accuracy_score(y_val, predictions)
print('Val Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_val, predictions)
cm = plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)
```

## Important Notes
1. Always use the same preprocessor that was fitted on the training data
2. Never fit the preprocessor on validation data (to avoid data leakage)
3. Make sure to:
   - Save your preprocessor after training
   - Use the same features in validation as in training
   - Apply identical preprocessing steps

## Complete Example
We've provided a complete implementation in `model_validation.py` that includes:
- A reusable validation function
- Proper preprocessing handling
- Confusion matrix plotting

You can use it like this:
```python
from model_validation import validate_model

# After training your model
accuracy, conf_matrix = validate_model(model, X_val, preprocessor)
```

This will handle all the preprocessing and validation steps in a clean, reusable way.