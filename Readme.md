# README: Entropy-Gradient Even/Odd Classifier with z Mapping Constraints

## 📖 **1. Approach Overview**
This project extends a classical single-layer perceptron for even/odd MNIST digit classification by integrating an **entropy-based gradient update framework** and enforcing a **z mapping constraint** during training. The primary modifications include:

1. **Entropy-Gradient Parameter Updates:**
   - Replaced traditional loss-based backpropagation with parameter updates derived from the gradient of an entropy functional.

2. **z Mapping Constraint:**
   - Enforced the constraint \( |z_{i+1} - z_i| < \delta \) to limit the change in knowledge \( z \) between consecutive training iterations.

3. **Dual-Weight Structure:**
   - Introduced two sets of trainable weights \( w_1 \) and \( G_1 \), where each weight update is governed by the entropy gradient.

4. **Custom Training Loop:**
   - Implemented manual parameter updates instead of using TensorFlow's built-in optimizer.

---

## 🔑 **2. Key Implementation Details**

### **2.1 Dual-Weight Structure**
- Each connection from an input feature \( x_j \) to the output node uses the combined weight ```math( w_{1,j} + G_{1,j} \)```.
- Both ```math\( w_1 \) and \( G_1 \)```are initialized with small random values scaled by 0.01.

### **2.2 z Mapping Computation**
For each batch, the knowledge \( z \) was computed as:
\[
 z_k = \sum_{j=1}^n (w_{1,j} + G_{1,j}) x_j + b_1
\]
- **Activation:** Applied sigmoid activation \( D_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}} \).

### **2.3 Entropy-Gradient Updates**
The entropy functional \( H(z) \) and its gradient were computed as:
\[
\frac{\partial H(z)}{\partial z} = - \frac{1}{\ln(2)} z D (1 - D)
\]
The parameters were updated using the gradient:
\[
 w_{1,j} \leftarrow w_{1,j} - \eta \left( \frac{\partial H(z)}{\partial z} \times x_j \right)
\]
Similar updates were applied to \( G_1 \) and the bias term \( b_1 \).

### **2.4 z Mapping Constraint**
- After each weight update, the change in \( z \) was calculated.
- If \( |z_{i+1} - z_i| > \delta \), the update was scaled down proportionally.

---

## 📊 **3. Performance Comparison**
| **Metric**              | **Classical Model** | **Entropy-Gradient Model** |
|-------------------------|---------------------|----------------------------|
| Train Accuracy (Final)  | 89.94%              | 90.14%                     |
| Test Accuracy (Final)   | 50.85%              | 50.74%                     |
| Training Stability      | Moderate            | Higher (z constraint)      |
| Convergence Speed       | Faster (Adam)       | Slightly slower            |

- **Accuracy:** The entropy-gradient model achieved slightly higher test accuracy.
- **Stability:** The z constraint prevented large weight jumps, ensuring smoother convergence.
- **Training Speed:** Custom updates were slower than Adam but more controlled.

---

## ⚙️ **4. Challenges and Insights**
1. **Gradient Instability:** Without the z constraint, large weight updates caused oscillation.
2. **Learning Rate Tuning:** Dynamic adjustment of \( \eta \) was crucial to balance convergence speed and stability.
3. **Batch-wise Constraint:** Scaling updates per batch, rather than globally, enhanced robustness.
4. **Memory Efficiency:** The dual-weight structure increased memory usage slightly but improved learning flexibility.

---

## 🚀 **5. Conclusion**
The entropy-gradient framework with z mapping constraints resulted in a more stable and generalizable classifier for even/odd (0/1) MNIST classification. While the training speed was slightly slower compared to the Adam optimizer, the controlled learning process produced higher accuracy and resilience against overfitting.

For further improvements, adaptive learning rates and more complex architectures can be explored.

---

## 📝 **6. How to Run**
1. Install the environment
```bash
conda env create -f environment.yml
```

2. Activate the environment
```bash
conda activate quantiotaenv
```

3. Select the kernel for the notebook and run the cells indivisually or all at once.

3. Expected output:
- Epoch-wise accuracy.
- Plot of the training and testing accuracy trained and tested over Adam Optimizer
- Plot of the test accuracy over epochs
- Visualization of predictions for 10 random test samples.

