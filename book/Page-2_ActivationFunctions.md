[Back to Index](./O-index.md)  [Previous.Page](./Page-1_Single_AND_MultiLayers_Perceptrons.md)  [Next.Page](./Page-3_MultiClassClassification.md)
In neural networks, **activation functions** introduce non-linearity to the model, allowing it to learn and represent more complex patterns and relationships. Without activation functions, the neural network would be a purely linear transformation, no more powerful than a single-layer perceptron, which would severely limit its ability to model real-world problems.

Activation functions operate on the neurons' output after the weighted sum of inputs has been computed, transforming the input signal before passing it to the next layer or producing the final output.

### Common Activation Functions

1. **Sigmoid (Logistic) Function**:
   \[
   \sigma(x) = \frac{1}{1 + e^{-x}}
   \]
   - **Range**: (0, 1)
   - **Use Case**: Often used in binary classification tasks and output layers.
   - **Pros**: Smooth gradient, outputs probability-like values.
   - **Cons**: Prone to vanishing gradient problem, especially in deep networks.

2. **Tanh (Hyperbolic Tangent)**:
   \[
   \tanh(x) = \frac{2}{1 + e^{-2x}} - 1
   \]
   - **Range**: (-1, 1)
   - **Use Case**: Hidden layers, useful when output needs to range between negative and positive values.
   - **Pros**: Zero-centered, which can make training faster.
   - **Cons**: Still suffers from the vanishing gradient problem.

3. **ReLU (Rectified Linear Unit)**:
   \[
   \text{ReLU}(x) = \max(0, x)
   \]
   - **Range**: [0, ∞)
   - **Use Case**: Widely used in hidden layers of deep neural networks due to its simplicity and effectiveness.
   - **Pros**: Computationally efficient, reduces the likelihood of vanishing gradients.
   - **Cons**: Can cause "dead neurons" where neurons stop updating during training (i.e., if inputs are always negative, gradient is zero).

4. **Leaky ReLU**:
   \[
   \text{Leaky ReLU}(x) = \begin{cases}
   x & \text{if } x > 0 \\
   \alpha x & \text{if } x \leq 0
   \end{cases}
   \]
   - **Range**: (-∞, ∞)
   - **Use Case**: Hidden layers where ReLU might fail due to dead neurons.
   - **Pros**: Avoids "dead neurons" by allowing small gradients for negative inputs (typically \(\alpha = 0.01\)).
   - **Cons**: Requires additional hyperparameter tuning for \(\alpha\).

5. **Softmax**:
   \[
   \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
   \]
   - **Range**: (0, 1) for each output, and all outputs sum to 1.
   - **Use Case**: Used in the output layer for multi-class classification tasks.
   - **Pros**: Converts logits to probabilities, useful in classification.
   - **Cons**: Computation can be expensive for large output spaces.

6. **ELU (Exponential Linear Unit)**:
   \[
   \text{ELU}(x) = \begin{cases}
   x & \text{if } x > 0 \\
   \alpha (e^x - 1) & \text{if } x \leq 0
   \end{cases}
   \]
   - **Range**: (-α, ∞)
   - **Use Case**: Like ReLU but with reduced "dead neuron" issue.
   - **Pros**: Non-zero mean activations which can help learning.
   - **Cons**: More computationally expensive due to the exponential operation.

7. **Swish**:
   \[
   \text{Swish}(x) = x \cdot \sigma(x)
   \]
   - **Range**: (-∞, ∞)
   - **Use Case**: General-purpose activation function.
   - **Pros**: Smooth and non-monotonic, empirically shown to improve performance in some deep learning architectures.
   - **Cons**: Slightly more computationally expensive than ReLU.

### Importance of Non-Linearity
The non-linearity introduced by these activation functions enables the network to approximate complex functions and capture intricate data patterns. Without them, a neural network would essentially be a series of linear transformations, which can't model complex, real-world relationships.

### Summary Table

| **Activation Function** | **Range**       | **Pros**                        | **Cons**                       | **Use Case**                   |
|-------------------------|-----------------|----------------------------------|---------------------------------|---------------------------------|
| Sigmoid                 | (0, 1)          | Probability-like output          | Vanishing gradient              | Binary classification output    |
| Tanh                    | (-1, 1)         | Zero-centered                    | Vanishing gradient              | Hidden layers                   |
| ReLU                    | [0, ∞)          | Computational efficiency         | Dead neurons                    | Deep hidden layers              |
| Leaky ReLU              | (-∞, ∞)         | Allows small negative gradients  | Hyperparameter tuning required  | Hidden layers with sparse data  |
| Softmax                 | (0, 1)          | Probability distribution output  | Expensive for large classes     | Multi-class classification      |
| ELU                     | (-α, ∞)         | Reduces dead neurons             | Computationally expensive       | Similar to ReLU but smoother    |
| Swish                   | (-∞, ∞)         | Smooth gradient, non-monotonic   | Slightly slower than ReLU       | Deep networks                   |

These activation functions are fundamental to how neural networks learn and generalize from data, making them crucial components in modern machine learning models.
[Back to Index](./O-index.md)  [Previous.Page](./Page-1_Single_AND_MultiLayers_Perceptrons.md)  [Next.Page](./Page-3_MultiClassClassification.md)
