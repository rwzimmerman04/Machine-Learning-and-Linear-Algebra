# Perceptron

> Study notes. Related: [SVM.md](../svm/SVM.md) (builds directly on the ideas here).

---

## 1. Big picture

- The perceptron (Rosenblatt) is a **linear classifier**: it learns a line (hyperplane) that separates two classes.
- It is the building block of the **multi-layer perceptron (MLP)**, i.e. a neural network.
- Intuition for the algorithm:
  1. Start with a random line.
  2. Check every point.
  3. If a point is misclassified, try to fix it. If not, move on.
- Simple, but powerful.

---

## 2. Setup and assumptions

The perceptron is for **classification**.

| Assumption | Meaning |
|---|---|
| $y \in \{-1, +1\}$ | Two classes, labeled $-1$ and $+1$ |
| Data is **linearly separable** (LS) | Some line/hyperplane $w^\top x = 0$ separates the two categories perfectly |

Each data point is $x = (x_1, x_2, \dots, x_d)$, and the model uses a weighted sum:

$$w^\top x = w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$

---

## 3. Hyperplanes and the vector w

The hyperplane is the set of points where the weighted sum is zero:

$$H = \{\, x \mid w^\top x = 0 \,\}$$

**Claim: $w$ is perpendicular (normal) to the hyperplane.**

Take any two points $x_1, x_2$ on $H$, so $w^\top x_1 = 0$ and $w^\top x_2 = 0$. Then

$$w^\top (x_2 - x_1) = w^\top x_2 - w^\top x_1 = 0 - 0 = 0$$

$x_2 - x_1$ is a vector lying *along* the hyperplane, and $w$ is orthogonal to it. Since this holds for any two points, $w$ is orthogonal to the whole hyperplane.

---

## 4. Classification rule

| Value | Meaning |
|---|---|
| $w^\top x = 0$ | $x$ is **on** the hyperplane |
| $w^\top x > 0$ | $x$ is on the **+** side, predict $+1$ |
| $w^\top x < 0$ | $x$ is on the **−** side, predict $-1$ |

**Correct vs. mistake.** Multiplying by the true label $y$ folds both cases into one test:

- $y \, w^\top x > 0$: **correct** (prediction and label agree in sign)
- $y \, w^\top x \le 0$: **mistake**

> This single inequality is what the algorithm checks. Remember it.


---

## 5. The perceptron algorithm

```
init  w := 0
for epoch = 1 ... T:
    for each data point (x_i, y_i) in D:
        if y_i * (w^T x_i) <= 0:      # mistake
            w := w + y_i * x_i        # fix it
```

The two cases of the update:

| Mistake type | Update |
|---|---|
| $y_i = +1$ but $w^\top x_i < 0$ | $w := w + x$ |
| $y_i = -1$ but $w^\top x_i > 0$ | $w := w - x$ |

Both are the single rule $w := w + y_i x_i$.

### Python version

```python
import numpy as np

def perceptron(X, y, epochs=100):
    """
    X: (N, d) array 
    y: (N,) array of -1/+1. 
       
    Returns weight vector w.
    """
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi) <= 0:   # mistake
                w += yi * xi         # update
                mistakes += 1
        if mistakes == 0:            # a full clean pass: converged
            break
    return w
```

---

## 6. Why the update works

**Case $y = +1$, but $w^\top x < 0$.** After the update, the new score on $x$ is

$$(w + x)^\top x = \underbrace{w^\top x}_{<0} + \underbrace{x^\top x}_{=\|x\|^2 \,>\, 0}$$

The score becomes **less negative, or now positive**.

**Case $y = -1$, but $w^\top x > 0$.** After the update ($w - x$):

$$(w - x)^\top x = \underbrace{w^\top x}_{>0} - \underbrace{\|x\|^2}_{>0}$$

The score becomes **less positive, or now negative**.

**Geometric picture.** Adding (or subtracting) $x$ rotates $w$ toward (or away from) $x$. The hyperplane always stays perpendicular to $w$, so it **pivots** to follow $w_{\text{new}}$.

```
 y = +1 mistake:   w_new = w + x    ->  w tilts toward x
 y = -1 mistake:   w_new = w - x    ->  w tilts away from x
```

> One update does not necessarily fix that point. It only moves in the right direction. The point may need several passes.

---

## 7. Margin and convergence

$\gamma$ (gamma) is the **margin**: roughly, how far the closest data point is from the separating hyperplane.

- Large $\gamma$: classes are well separated, so it is easy.
- Small $\gamma$: classes are barely separable, so it is hard.

**Perceptron mistake bound:**

$$\text{number of mistakes} \;\propto\; \frac{1}{\gamma^2}$$

(The precise form is $\le R^2/\gamma^2$, where $R$ is the max norm of any $x$. With data scaled so $R = 1$, this is $1/\gamma^2$.)

So if the data is linearly separable, the perceptron makes a *finite* number of mistakes and **converges**. A bigger margin means faster convergence.

> The margin idea is exactly what the [SVM](../svm/SVM.md) is built around: instead of finding *any* separating hyperplane, find the one with the *largest* margin.

---

## 8. Limitations: XOR and the AI winter

**XOR problem.** Four points on a grid:

```
 y=1 |  x    o
 y=0 |  o    x
     +-----------
        0    1
```

There is **no way to use a single straight line to divide all the x's from all the o's.**

- Minsky showed the perceptron could not handle this, which contributed to the **AI winter** (roughly 20-30 years).
- More generally, if the data is **not linearly separable, the perceptron will NOT converge.** It keeps updating forever.

---

## 9. Logistic regression: the smooth analog

Logistic regression is a **smooth analog of the perceptron**. It is still a linear model, but it "squishes" the score between 0 and 1 with the sigmoid $\sigma$.

$$P(y = 1 \mid w, x) = \sigma(w^\top x)$$

Compactly for $y \in \{-1, +1\}$:

$$P(y \mid x, w) = \sigma(y \, w^\top x)$$

$$P(y = -1 \mid x, w) = \sigma(-w^\top x) = 1 - \sigma(w^\top x)$$

**Connection to the perceptron.** Scale the input: $\sigma(\alpha z)$ with $z = w^\top x$. The slope at $z = 0$ grows with $\alpha$. As $\alpha \to \infty$, the sigmoid becomes a **step function**, which is the perceptron's hard decision.

| | Perceptron | Logistic regression |
|---|---|---|
| Output | hard $\pm 1$ (step) | smooth probability in $(0,1)$ |
| Non-separable data | **does NOT converge** | **WILL converge** |

---

## 10. Recap: least squares

Related background from the same notes. Least squares fits $y \approx Xw$.

With data matrix $X$ ($N \times d$, where $d \ll N$):

$$\hat{w} = (X^\top X)^{-1} X^\top y = X^{+} y$$

- $X^{+}$ is the **pseudo-inverse** of $X$.
- $X^\top X$ is only $d \times d$, so it is much more likely to be **full rank and invertible**.
- Informally: $y \approx Xw \Rightarrow \hat{w} \approx X^{-1} y$, but $X$ is not square, so we use $X^{+}$.
