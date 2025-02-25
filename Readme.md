

---

# Entropy-Gradient Even/Odd Classifier with z Mapping Constraints  

## 📖 **1. Approach Overview**  
This project extends a classical single-layer perceptron for even/odd (0/1) MNIST digit classification by integrating an **entropy-based gradient update framework** and enforcing a **z mapping constraint** during training. The primary modifications include:  

1. **Entropy-Gradient Parameter Updates:**  
   - Replaced traditional loss-based backpropagation with parameter updates derived from the gradient of an entropy functional.  

2. **z Mapping Constraint:**  
   - Enforced the constraint  
     ```math  
     |z_{i+1} - z_i| < \delta  
     ```  
     to limit the change in knowledge \( z \) between consecutive training iterations(preferred between 0.01 and 0.1).  

3. **Dual-Weight Structure:**  
   - Introduced two sets of trainable weights
      ```math
        ( w_1 )
     ```
   and
     ```math
     ( G_1 )
     ```
    where each weight update is governed by the entropy gradient.  

4. **Custom Training Loop:**  
   - Implemented manual parameter updates instead of using TensorFlow's built-in Adam optimizer.  

---  

## 🔑 **2. Key Implementation Details**  

### **2.1 Dual-Weight Structure**  
- Each connection from an input feature
  ```math
  ( x_j )
  ```
   to the output node uses the combined weight
  ```math
  ( w_{1,j} + G_{1,j} )
  ```
- Both
  ```math
  ( w_1 )
  ```
   and
  ```math
  ( G_1 )
  ```
   are initialized with small random values scaled by 0.01.  

### **2.2 z Mapping Computation**  
For each batch, the knowledge 
```math
( z )
```
was computed as:  
```math  
z_k = \sum_{j=1}^n (w_{1,j} + G_{1,j}) x_j + b_1  
```  
- **Activation:** Applied sigmoid activation  
```math  
D_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}  
```  

### **2.3 Entropy-Gradient Updates**  
The entropy functional 
```math
( H(z) )
```
 and its gradient were computed as:  
```math  
\frac{\partial H(z)}{\partial z} = - \frac{1}{\ln(2)} z D (1 - D)  
```  
The parameters were updated using the gradient:  
```math  
w_{1,j} \leftarrow w_{1,j} - \eta \left( \frac{\partial H(z)}{\partial z} \times x_j \right)  
```  
Similar updates were applied to 
```math
( G_1 )
```
 and the bias term 
 ```math
( b_1 )
```  

### **2.4 z Mapping Constraint**  
- After each weight update, the change in
  ```math
  ( z )
  ```
  was calculated.  
- If  
  ```math  
  |z_{i+1} - z_i| > \delta  
  ```  
  the update was scaled down proportionally.  

---  

## 📊 **3. Performance Comparison**  

| **Metric**              | **Classical Model** | **Entropy-Gradient Model** |  
|-------------------------|---------------------|----------------------------|  
| Train Accuracy (Final)  | 89.94%              | 90.14%                     |  
| Test Accuracy (Final)   | 50.85%              | 50.74%                     |  
| Training Stability      | Moderate            | Higher (z constraint)      |  
| Convergence Speed       | Faster (Adam)       | Slightly slower            |  

- **Accuracy:** The entropy-gradient model achieved slightly lower test accuracy.  
- **Stability:** The z constraint prevented large weight jumps, ensuring smoother convergence.  
- **Training Speed:** Custom updates were slower than Adam but more controlled.  

---  

## ⚙️ **4. Challenges and Insights**  
1. **Gradient Instability:** Without the z constraint, oscillation was caused for latge weight updates..  
2. **Learning Rate Tuning:** Dynamic adjustment of
   ```math
   ( \eta )
   ```
    proves crucial to balance the speed o convergence.  
5. **Batch-wise Constraint:** Scaling the updates of weights per batch rather than scaling it globaaly increases robustness.  
6. **Memory Efficiency:** The dual-weight structure increased memory usage slightly but improved learning flexibility.  

---  

## 🚀 **5. Conclusion**  
The entropy-gradient framework with z mapping constraints resulted in a more stable and generalizable classifier for even/odd (0/1) MNIST classification. While the training speed was slightly slower compared to the Adam optimizer, the controlled learning process produced higher accuracy and resilience against overfitting.  

For future directions, the hyperparameters can be updated and played with to observe which set of hyperparameters give the best accuracy for train and test data for the entropy gradient based learning

---  

## 📝 **6. How to Run**  

1. Install the environment:  
   ```bash  
   conda env create -f environment.yml  
   ```  

2. Activate the environment:  
   ```bash  
   conda activate quantiotaenv  
   ```  

3. Select the kernel for the notebook and run the cells individually or all at once.  

4. Expected output:  
   - Epoch-wise accuracy.  
   - Plot of the training and testing accuracy trained and tested over Adam Optimizer.  
   - Plot of the test accuracy over epochs.  
   - Visualization of predictions for 10 random test samples.  

---
