[Back to Index](./O-index.md)  [Previous.Page](./Page-4_MutuallyExclusiveClasses.md)  [Next.Page](./Page-6_Backpropagation.md)

### Cost Functions and Gradient Descent

**Cost functions** and **gradient descent** are fundamental concepts in machine learning and optimization, particularly in training models like neural networks. Together, they help the model learn the best parameters (such as weights and biases) to minimize errors and make accurate predictions.

---

### 1. **Cost Functions** (or Loss Functions)

A **cost function** (also called a **loss function** or **objective function**) measures the error between the predicted output of a model and the actual output. It tells us how well the model is performing. The goal of training is to minimize this cost function by adjusting the model’s parameters.

#### Common Cost Functions

1. **Mean Squared Error (MSE)**: Used for regression tasks.
   - Measures the average squared difference between the predicted values and the actual values.
   - Formula:
     \[
     \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
     \]
     where \( y^{(i)} \) is the actual value, \( \hat{y}^{(i)} \) is the predicted value, and \( m \) is the number of data points.

2. **Cross-Entropy Loss (Log Loss)**: Used for classification tasks.
   - Measures the difference between two probability distributions: the true distribution (actual labels) and the predicted distribution (model’s output).
   - Formula for binary classification:
     \[
     \text{Cross-Entropy Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
     \]
     where \( \hat{y}^{(i)} \) is the predicted probability, and \( y^{(i)} \) is the actual label (0 or 1).

3. **Hinge Loss**: Used for "maximum-margin" classifiers like Support Vector Machines (SVMs).
   - Formula:
     \[
     \text{Hinge Loss} = \frac{1}{m} \sum_{i=1}^{m} \max(0, 1 - y^{(i)} \hat{y}^{(i)})
     \]
     where \( y^{(i)} \) is the actual label and \( \hat{y}^{(i)} \) is the predicted value.

#### Purpose of the Cost Function:
- The cost function quantifies the error of a model.
- It provides a way to assess how well the model fits the data.
- The goal during training is to minimize the value of the cost function by adjusting the model’s parameters.

---

### 2. **Gradient Descent**

**Gradient Descent** is an optimization algorithm used to minimize the cost function by iteratively adjusting the model’s parameters (such as weights and biases). It computes the gradients of the cost function with respect to the parameters and updates the parameters in the direction that reduces the cost.

#### How Gradient Descent Works:

1. **Initialization**: Start with random values for the parameters (weights and bias).
2. **Compute Gradients**: Calculate the gradient of the cost function with respect to the parameters. The gradient is the vector of partial derivatives that points in the direction of the steepest increase in the cost function.
3. **Update Parameters**: Move the parameters in the opposite direction of the gradient to reduce the cost. The update rule for the weights and biases is:
   \[
   w_j := w_j - \eta \frac{\partial J(w)}{\partial w_j}
   \]
   \[
   b := b - \eta \frac{\partial J(b)}{\partial b}
   \]
   where \( \eta \) is the **learning rate**, and \( J(w) \) is the cost function.
4. **Repeat**: Continue updating the parameters until the cost function converges to a minimum (or when the changes become very small).

#### Example of Gradient Descent:

Consider a simple linear regression model, where the cost function is the Mean Squared Error (MSE):
\[
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
\]
Here, \( w \) and \( b \) are the parameters (weight and bias), and the goal is to minimize \( J(w, b) \).

- **Gradient with respect to the weight** \( w \):
  \[
  \frac{\partial J(w)}{\partial w} = -\frac{2}{m} \sum_{i=1}^{m} x^{(i)} (y^{(i)} - \hat{y}^{(i)})
  \]
- **Gradient with respect to the bias** \( b \):
  \[
  \frac{\partial J(b)}{\partial b} = -\frac{2}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})
  \]

After calculating the gradients, you update the weight \( w \) and bias \( b \) using the update rules above. This process is repeated until the cost function reaches its minimum.

---

### Types of Gradient Descent

1. **Batch Gradient Descent**:
   - Uses the entire dataset to compute the gradients.
   - The update is more stable but slower for large datasets.
   - Formula:
     \[
     w := w - \eta \frac{1}{m} \sum_{i=1}^{m} \frac{\partial J}{\partial w}
     \]
     where \( m \) is the total number of data points.

2. **Stochastic Gradient Descent (SGD)**:
   - Uses a single data point at each iteration to compute the gradient.
   - Faster but can have high variance in the updates, causing it to bounce around the minimum.
   - Formula:
     \[
     w := w - \eta \frac{\partial J^{(i)}}{\partial w}
     \]
     where \( J^{(i)} \) is the cost for a single data point.

3. **Mini-Batch Gradient Descent**:
   - Uses a small batch of data points (usually 32 or 64) to compute the gradient.
   - Provides a balance between the stability of batch gradient descent and the speed of stochastic gradient descent.

---

### Example: Linear Regression with Gradient Descent

Consider a simple linear regression problem where we want to fit a line to the data. The model is defined as:
\[
\hat{y} = w x + b
\]
where \( x \) is the input, \( w \) is the weight, and \( b \) is the bias.

1. **Cost Function (MSE)**:
   \[
   J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - (w x^{(i)} + b))^2
   \]
   The goal is to minimize this cost function by adjusting \( w \) and \( b \).

2. **Gradient Descent Steps**:
   - Compute the gradient of the cost function with respect to \( w \) and \( b \).
   - Update \( w \) and \( b \) in the direction that minimizes the cost function.

After many iterations, the values of \( w \) and \( b \) will converge to the values that minimize the cost function, yielding the best-fitting line.

---

### Example: Logistic Regression with Gradient Descent

In logistic regression, the cost function is the **cross-entropy loss**, and the goal is to minimize it. The update rule for the weights is similar to that in linear regression, but the model’s output is a probability:
\[
\hat{y} = \frac{1}{1 + e^{-(w x + b)}}
\]
Here, \( \hat{y} \) is the predicted probability, and the parameters \( w \) and \( b \) are updated using gradient descent.

---

### Summary
- **Cost functions** measure the error between predicted and actual values and provide a way to evaluate how well a model is performing.
- **Gradient Descent** is an optimization algorithm that adjusts model parameters to minimize the cost function.
- There are different types of gradient descent (batch, stochastic, mini-batch) that trade off between speed and accuracy.

Together, cost functions and gradient descent enable machine learning models to learn and improve by minimizing errors.

[Back to Index](./O-index.md)  [Previous.Page](./Page-4_MutuallyExclusiveClasses.md)  [Next.Page](./Page-6_Backpropagation.md)
