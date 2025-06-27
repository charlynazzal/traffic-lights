# Traffic Light Classification Project

This project aims to build a machine learning model to classify traffic light signals (red, yellow, green) based on the "Traffic Light Signals at Various Intersections" dataset from Kaggle.

## Methodology and Model Iteration

The model was developed through a systematic, iterative process focused on maximizing performance on the small dataset.

1.  **Initial Baseline:** A baseline neural network (`128 -> 64` layers) was established. It achieved an initial accuracy of **51.11%** but showed clear signs of overfitting, with the validation loss increasing steadily during training.
2.  **Addressing Overfitting (Dropout):** To combat overfitting, dropout layers (`p=0.5`) were introduced. This improved model generalization and increased the test accuracy to **55.56%**.
3.  **Architecture Tuning:** A hypothesis was formed that the model might be too complex for the dataset. A simpler architecture (`64 -> 32`) was tested, but this resulted in underfitting and a lower accuracy of `46.67%`, proving the previous architecture was more effective.
4.  **Hyperparameter Tuning (Learning Rate):** With the optimal architecture (`128 -> 64` with dropout) re-established, a systematic search for the best learning rate was conducted. The initial rate of `0.001` was compared against several others. It was discovered that a learning rate of **0.01** provided a significant performance boost.

### Final Results and Analysis

The final, optimized model achieved a **test accuracy of 66.67%**.

A more detailed analysis was performed to understand the model's specific strengths and weaknesses:

**Classification Report:**
```
              precision    recall  f1-score   support
       Green       0.57      0.57      0.57        14
         Red       0.57      0.53      0.55        15
      Yellow       0.88      0.82      0.85        16
    --------------------------------------------------
    accuracy                           0.67        45
   macro avg       0.67      0.64      0.66        45
weighted avg       0.68      0.67      0.67        45
```

**Confusion Matrix:**
```
      Predicted:
      G   R   Y
G [[ 8,  6,  0],
R  [ 4,  8,  2],
Y  [ 2,  1, 14]]
```
*(Note: Rows are True Labels, Columns are Predicted Labels)*

**Analysis:**
The model performs best when identifying the "Yellow" state, showing high precision and recall. However, the confusion matrix reveals a critical weakness: the model frequently misclassifies "Red" lights as "Green" (4 instances) and vice-versa (6 instances). In a real-world application, such errors would be unacceptable and dangerous. This analysis highlights that while the overall accuracy is a useful starting point, a deeper dive into class-specific metrics is crucial for evaluating the true viability of a model. 