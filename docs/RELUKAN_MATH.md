# ReLU-KAN Mathematical Formulation

## ReLU-KAN Layer

### Basis Functions

Given input $x \in \mathbb{R}^{n_{in}}$ and parameters:
- $\mathbf{t}_{pos}, \mathbf{t}_{neg} \in \mathbb{R}^{n_{in} \times k}$ (knot positions)
- $\mathbf{a}_{pos}, \mathbf{a}_{neg} \in \mathbb{R}^{n_{in} \times k}$ (coefficients)
- $\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}$ (weight matrix)

Define bidirectional ReLU basis functions:

$$\phi_{pos}^{(i)}(x_i) = \sum_{j=1}^{k} \text{ReLU}(a_{pos}^{(i,j)} \cdot (x_i - t_{pos}^{(i,j)}))$$

$$\phi_{neg}^{(i)}(x_i) = \sum_{j=1}^{k} \text{ReLU}(-a_{neg}^{(i,j)} \cdot (x_i - t_{neg}^{(i,j)}))$$

### Feature Transformation

Softmin approximation of $\min(\phi_{pos}, \phi_{neg})$:

$$\phi_i(x_i) = \text{softmin}(\phi_{pos}^{(i)}(x_i), \phi_{neg}^{(i)}(x_i)) = -\frac{1}{\beta}\log\left(\sum_{m \in \{pos,neg\}} e^{-\beta \phi_m^{(i)}(x_i)}\right)$$

where $\beta = 100$ controls approximation sharpness.

### Output

$$y = \phi(x) \mathbf{W}^T, \quad \phi(x) = [\phi_1(x_1), \ldots, \phi_{n_{in}}(x_{n_{in}})]$$

---

## Interval Penalization for Continual Learning

Let $\theta^{(t-1)}$ denote parameters after task $t-1$, and $\mathcal{D}^{(t-1)}$ the input domain of task $t-1$.

### Regularization Terms

**1. Output Variance Loss**

$$\mathcal{L}_{var} = \mathbb{E}_{x \sim \mathcal{D}^{(t)}} \left[\text{Var}(y(x))\right]$$

Encourages compact and stable output representations.

**2. Knot Displacement Penalty**

For each input dimension $i$ with range $[x_i^{min}, x_i^{max}]$ from $\mathcal{D}^{(t-1)}$:

$$\mathcal{L}_{knot} = \sum_{i=1}^{n_{in}} \sum_{j=1}^{k} \mathbb{1}_{[x_i^{min}, x_i^{max}]}(t_{pos}^{(i,j)}) \cdot \|t_{pos}^{(i,j)} - t_{pos}^{(i,j), (t-1)}\|^2$$
$$+ \sum_{i=1}^{n_{in}} \sum_{j=1}^{k} \mathbb{1}_{[x_i^{min}, x_i^{max}]}(t_{neg}^{(i,j)}) \cdot \|t_{neg}^{(i,j)} - t_{neg}^{(i,j), (t-1)}\|^2$$

where $\mathbb{1}_{[a,b]}(t) = 1$ if $t \in [a,b]$, else $0$.

**3. Boundary Consistency Loss**

$$\mathcal{L}_{boundary} = \sum_{j=1}^{n_{out}} \left[\|y_j^{(t-1)}(x^{min}) - y_j^{(t)}(x^{min})\|^2 + \|y_j^{(t-1)}(x^{max}) - y_j^{(t)}(x^{max})\|^2\right]$$

Preserves layer output values at domain boundaries.

**4. Output Interval Alignment Loss**

For output ranges $[y_j^{min, (t-1)}, y_j^{max, (t-1)}]$ from task $t-1$:

$$c_j^{(t-1)} = \frac{y_j^{min, (t-1)} + y_j^{max, (t-1)}}{2}, \quad w_j^{(t-1)} = y_j^{max, (t-1)} - y_j^{min, (t-1)}$$

$$c_j^{(t)} = \frac{\min_x y_j^{(t)}(x) + \max_x y_j^{(t)}(x)}{2}$$

$$\mathcal{L}_{align} = \sum_{j=1}^{n_{out}} \frac{\|c_j^{(t)} - c_j^{(t-1)}\|^2}{w_j^{(t-1)} + \epsilon}$$

Maintains output stability across tasks.

### Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{var} \mathcal{L}_{var} + \lambda_{knot} \mathcal{L}_{knot} + \lambda_{boundary} \mathcal{L}_{boundary} + \lambda_{align} \mathcal{L}_{align}$$

where $\mathcal{L}_{task}$ is the supervised task loss (e.g., cross-entropy).
