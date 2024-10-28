# Hyperparameter Tuning

By:
- **Aviel Avshalumov**
- **Ali Ghaziasgar**
- **Aldo Delgado**

## 1. What is Hyperparameter Tuning?

### Definition of Hyperparameter Tuning:
Hyperparameter tuning is the process of selecting the optimal configuration of hyperparameters that allow a machine learning model to perform at its best. Since hyperparameters are settings that are not learned during training but are set before the training process, tuning involves experimenting with different values to find the combination that optimizes the model’s performance.

In essence, hyperparameter tuning adjusts the "knobs" of the machine learning algorithm (e.g., learning rate, batch size, or regularization strength) to balance between underfitting (model too simple) and overfitting (model too complex).

### Definition of Hyperparameters:
Hyperparameters are settings that define how a machine learning model learns. Unlike model parameters (e.g., weights in neural networks), hyperparameters must be specified before training begins. They control aspects like the learning process and the architecture of the model, ultimately shaping the model’s performance.

For example, in a neural network:
- **Model Parameters**: Weights connecting neurons that are learned during training.
- **Hyperparameters**: Settings like learning rate, batch size, number of hidden layers, etc., which must be set before training starts.

### Role in Model Training:
Hyperparameters control critical aspects of how a model learns from data. Choosing the right hyperparameters can significantly influence the model’s convergence, speed of learning, and generalization to new data. Examples include:
- **Learning rate**: Determines the step size at each iteration in minimizing the loss function.
- **Batch size**: Defines how many training examples are used in each iteration.
- **Number of hidden layers**: Influences the complexity and capacity of neural networks.

Correctly choosing hyperparameters is crucial because they significantly affect how well the model generalizes to unseen data, preventing issues like underfitting or overfitting.

### When is Hyperparameter Tuning Done?
Hyperparameter tuning is typically performed after the initial model is built, during the validation phase. Once the model architecture is decided, tuning can improve the model’s performance on a validation set, helping ensure it generalizes well to new data.

### How is Hyperparameter Tuning Done?
Several methods are commonly used to tune hyperparameters:
- **Grid Search**: An exhaustive search over a predefined range of hyperparameters.
- **Random Search**: Randomly sampling from a range of hyperparameters, which is often faster than grid search.
- **Bayesian Optimization**: A more advanced method that models the hyperparameter space probabilistically to find the best settings with fewer evaluations.

### Examples of Hyperparameters in Common Algorithms:

| **Algorithm**            | **Hyperparameters**                                                                 |
|--------------------------|-------------------------------------------------------------------------------------|
| **Neural Networks**       | - Learning rate<br>- Batch size<br>- Number of hidden layers<br>- Dropout rate      |
| **Decision Trees**        | - Maximum depth<br>- Minimum samples per split<br>- Minimum samples per leaf        |
| **Support Vector Machines (SVM)** | - Kernel type (linear, polynomial, RBF)<br>- Regularization parameter (C)<br>- Gamma (for RBF kernel) |

By effectively tuning hyperparameters, you can balance model complexity and performance, improving predictive power while avoiding overfitting or underfitting.


---

## 2. Why is Hyperparameter Tuning Important?

### Optimization of Model Performance:

Tuning hyperparameters is critical for optimizing a model’s performance. The proper combination of hyperparameters can improve both accuracy and generalization ability, allowing the model to make reliable predictions on unseen data. For example, a well-tuned model can avoid common issues like:

- **Overfitting**: The model is too complex (e.g., too many layers or a low regularization penalty), learning the noise in the training data instead of the underlying patterns. 
- **Underfitting**: The model is too simple (e.g., shallow decision trees or an insufficient number of hidden layers), failing to capture important patterns in the data. 

### Avoiding Overfitting/Underfitting:

- **Overfitting** occurs when the model performs well on training data but poorly on unseen data. This can happen if hyperparameters like the number of hidden layers or tree depth are set too high, making the model overly complex. 
- **Underfitting** happens when the model is too simple to capture the data’s complexity, leading to poor performance on both training and test data. 

Effective hyperparameter tuning helps avoid these extremes by ensuring a model is just complex enough to capture relevant patterns while still being generalizable to new data. 

### Efficiency and Resource Management:

Hyperparameter tuning also impacts the computational efficiency of a model. Certain hyperparameters, such as batch size or model complexity, can directly affect the time and memory requirements during training. For instance: 

- A very small batch size may lead to faster convergence but higher memory usage. 
- A very large decision tree may take longer to train and require more memory. 

Balancing hyperparameters can optimize resource use, allowing models to be trained efficiently without sacrificing performance.


## 3. Bayesian Optimization for Hyperparameter Tuning

### TPE Estimator:
The **Tree-structured Parzen Estimator (TPE)** is a specific variant of Bayesian optimization used in hyperparameter tuning. Instead of using Gaussian processes like traditional Bayesian optimization, TPE models the likelihood of a hyperparameter configuration being optimal by dividing the results into two distributions:
1. One distribution models the hyperparameter values that are associated with the best results.
2. The other distribution models the remaining hyperparameter values.

TPE focuses on the hyperparameters likely to yield better results, making it a more efficient and flexible choice for hyperparameter optimization, especially for high-dimensional hyperparameter spaces.

### Pros and Cons of TPE:
#### Pros:
- **Efficiency**: Like Bayesian optimization, TPE requires fewer evaluations compared to traditional grid or random search, making it faster to find optimal hyperparameter values.
- **Scalability**: TPE scales better for higher-dimensional hyperparameter spaces and can work effectively with complex hyperparameter configurations.

#### Cons:
- **Complexity**: TPE introduces more complexity than simpler methods such as grid or random search.
- **Requires Proper Configuration**: While TPE is flexible, it requires careful configuration and can be slow if not properly tuned, particularly in cases with a very large search space.

---

### Bayesian Optimization Overview:
Bayesian Optimization builds a probabilistic model (usually a Gaussian process) of the objective function and uses this model to choose hyperparameter values intelligently. This method strikes a balance between **exploration** (trying new values) and **exploitation** (focusing on the best-performing values).

### How Bayesian Optimization Works:
1. **Surrogate Model**: A probabilistic surrogate model approximates the objective function based on past evaluations.
2. **Acquisition Function**: Determines the next hyperparameter set to evaluate by balancing exploration and exploitation. Common acquisition functions include **Expected Improvement** and **Upper Confidence Bound**.
3. **Iterative Process**: The model evaluates new hyperparameters, updates the surrogate model, and repeats the process until convergence or the evaluation limit is reached.

### Pros and Cons of Bayesian Optimization:
#### Pros:
- **Efficiency**: Bayesian optimization is more efficient than both grid and random search, especially in high-dimensional hyperparameter spaces. It finds better hyperparameters with fewer evaluations.
- **Exploration-Exploitation Tradeoff**: By balancing exploration and exploitation, it focuses on finding the best hyperparameters more intelligently.

#### Cons:
- **Complexity**: It is more complex to implement than grid or random search, requiring knowledge of probabilistic models.
- **Slower for High Dimensions**: Although efficient, Bayesian optimization can be slow in extremely high-dimensional hyperparameter spaces because of the time required to update the surrogate model.

### Example Use Case:
For deep learning models like neural networks, Bayesian optimization can efficiently adjust hyperparameters such as the **learning rate**, **dropout rate**, and **batch size**. This method can quickly narrow down the best hyperparameters without requiring an exhaustive search, leading to more optimal models with fewer evaluations.

---

### You can continue here :)

# Hyperparameter Optimization Methods

## Grid Search
- **Description**: Searches through a predefined list of hyperparameters to find the best combination.
- **Suitability**: Suitable for a small number of hyperparameters.
- **Example**: For 3 hyperparameters with 4 values each, the total combinations are \(4 \times 4 \times 4 = 64\).

### Financial ML Applications:
- Used for basic trading strategies, backtesting specific parameter ranges.
- Easier to understand; used in risk management and regulatory compliance.

---

## Random Search
- **Description**: Searches random combinations of hyperparameters.
- **Suitability**: Recommended for large parameter spaces due to its efficiency.

### Financial ML Applications:
- Used in factor model optimization, portfolio weight optimization (modern portfolio optimization), daily rebalancing.

---

## TPE Estimator (Tree-structured Parzen Estimator)
- **Description**: A sequential model-based optimization (SMBO) approach that models the probability of good and bad outcomes.
  - A form of Bayesian optimization method; a probabilistic model.
  - Models distribution of hyperparameters (e.g. Gaussian).
  - Uses an objective function (accuracy, loss, etc.).

### Functionality:
- Estimates expected improvement to existing combinations for each possible set.
- For complex models, it will need to have more evaluations to better predict the performance of hyperparameters.

### Financial ML Applications:
- Used for sophisticated ML-based trading strategies, real-time hyperparameter changes, factor selection, risk model optimization, and modeling changes in market regime.

---

## Comparison of Hyperparameter Optimization Methods

| Hyperparameter Method | Pros                                                                                   | Cons                                                             | On Level                                   |
|-----------------------|----------------------------------------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------|
| **Grid Search**       | - Simple to understand<br>- All combinations are tested<br>- Finds the best combination in the provided list<br>- Deterministic (reproducible results) | - Extensive computations for large datasets<br>- Not suitable for complex models<br>- Ignores importance of different hyperparameters | - Exponential complexity; Number of combinations grows exponentially<br>- O(n^d) where n is the number of values per hyperparameter, d is the number of hyperparameters |
| **Random Search**     | - Fewer computations than Grid Search<br>- More efficient for a high number of hyperparameters | - May not find the optimal combination of hyperparameters<br>- Not reproducible results | - Linear complexity; Linear growth of combinations<br>- Faster than Grid Search<br>- O(k) where k is the number of iterations |
| **TPE**               | - Fewer calculations than Random Search<br>- Adaptive search based on expected improvement<br>- Faster for complex models and high number of hyperparameters<br>- Works well with conditional parameters<br>- Usually finds good solutions with fewer iterations. | - More complex to implement<br>- Requires sufficient prior tests on different hyperparameters (initial random exploration)<br>- May not find the global optima<br>- May overfit | - Log-linear complexity<br>- O(k * log(k)) where k is the number of iterations<br>- Exploration strategy based on the expected improvement probability<br>- Faster than Random Search |

---

## Additional Factors Affecting Speed
Other factors that affect the speed include:
- The cost of evaluating each configuration
- The dimensionality of the search space
- Implementation details
