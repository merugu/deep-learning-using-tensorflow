[Back to Index](./O-index) [Previous.Page](./Page-2_ActivationFunctions.md)  [Next.Page](./Page-4_MutuallyExclusiveClasses.md)

**Multiclass classification** is a type of machine learning problem where the task is to categorize instances into one of three or more classes (categories). Unlike binary classification, which deals with only two possible outcomes, multiclass classification involves multiple possible outcomes for a given input.

### Examples of Multiclass Classification Problems:
1. **Handwritten Digit Recognition**: Classifying images of handwritten digits (0–9) into one of 10 classes.
2. **Iris Flower Classification**: Predicting the species of an iris flower (Setosa, Versicolor, or Virginica) based on its features (sepal and petal length/width).
3. **Animal Classification**: Classifying images of animals into categories like "cat," "dog," "bird," etc.

### Common Techniques for Multiclass Classification

1. **One-vs-Rest (OvR) or One-vs-All (OvA)**:
   - In this strategy, for \( N \) classes, \( N \) separate binary classifiers are trained. Each classifier predicts whether an instance belongs to a specific class or not. The class with the highest score (probability or margin) from these classifiers is chosen as the final prediction.
   - Example: If you are classifying images into 3 classes (cat, dog, bird), you would train 3 binary classifiers: one for "cat vs. rest," one for "dog vs. rest," and one for "bird vs. rest."

2. **One-vs-One (OvO)**:
   - In this approach, \( N \times (N - 1) / 2 \) binary classifiers are trained, each distinguishing between a pair of classes. Each classifier handles a different pair of classes, and during prediction, a majority vote or voting scheme is used to determine the final class.
   - Example: With 3 classes (cat, dog, bird), you train 3 binary classifiers: "cat vs. dog," "cat vs. bird," and "dog vs. bird." The majority voting determines the final prediction.

3. **Softmax (used in Neural Networks)**:
   - Softmax is a generalization of the sigmoid activation function used for multiclass classification. It converts raw output scores (logits) into probabilities across multiple classes. The class with the highest probability is selected as the prediction.
   - Formula:
     \[
     \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
     \]
     where \( z_i \) is the raw score (logit) for class \( i \), and the denominator sums over all classes.

4. **Decision Trees and Random Forests**:
   - These models are naturally capable of handling multiclass classification without needing special treatment like OvR or OvO. A decision tree splits the data at each node based on the most informative feature, leading to a specific class at the leaf nodes. Random forests build multiple decision trees and take the majority vote from those trees.

5. **k-Nearest Neighbors (k-NN)**:
   - k-NN is a distance-based algorithm where the class of a new instance is determined by the majority class among its \( k \) nearest neighbors in the training data. It's naturally multiclass without needing additional strategies.

### Evaluation Metrics for Multiclass Classification

1. **Confusion Matrix**:
   - A confusion matrix for multiclass classification is a square matrix where each row represents the true class, and each column represents the predicted class. Diagonal values indicate correct classifications, and off-diagonal values indicate misclassifications.
   
2. **Precision, Recall, F1-Score** (per class):
   - For each class, precision, recall, and F1-score can be calculated as in binary classification, and then an average can be taken across all classes.
   - **Micro-average**: Takes into account the total true positives, false negatives, and false positives across all classes.
   - **Macro-average**: Averages precision/recall/F1-score across all classes equally, regardless of class frequency.

3. **Accuracy**:
   - The proportion of correct predictions over all predictions:
     \[
     \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
     \]

4. **Weighted Metrics**:
   - A weighted average of precision, recall, or F1-score where each class is weighted by its support (the number of true instances in that class).

### Challenges in Multiclass Classification
1. **Class Imbalance**:
   - In some cases, certain classes may have significantly more examples than others, leading to biased models that favor the majority classes.
   - Solutions include oversampling minority classes, undersampling majority classes, or using cost-sensitive learning.

2. **Complexity**:
   - As the number of classes increases, the classification task becomes more challenging due to increased variance and complexity in the decision boundaries.

### Summary
- **Multiclass classification** deals with problems where there are more than two classes.
- Techniques include **One-vs-Rest**, **One-vs-One**, and methods like **Softmax** in neural networks.
- Models like **decision trees**, **random forests**, and **k-NN** handle multiclass classification natively.
- Evaluation is done using metrics like accuracy, precision, recall, and F1-score, along with confusion matrices.

[Back to Index](./O-index)  [Previous.Page](./Page-2_ActivationFunctions.md)  [Next.Page](./Page-4_MutuallyExclusiveClasses.md)
