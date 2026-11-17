# Variance, Covariance, and Correlation

## Notation

| Symbol | Meaning |
|---|---|
| $m$ | Number of observations (rows) |
| $n$ | Number of features (columns) |
| $X \in \mathbb{R}^{m\times n}$ | Data matrix, one observation per row |
| $x_j$ | Column $j$ of $X$ (all observations of feature $j$) |
| $\mu_j$ | Mean of feature $j$ |
| $\mu \in \mathbb{R}^n$ | Vector of all feature means |
| $X_c = X - \mathbf{1}\mu^T$ | Centered data matrix |

## Mean

$$
\mu_j = \frac{1}{m}\sum_{i=1}^{m} x_{ij}
$$

The mean is the center of mass of a feature. **Centering** means subtracting it, so every
feature has mean 0 and all that remains is how the data *varies*.

## Variance

Variance measures how far one feature spreads around its mean:

$$
\operatorname{var}(x_j) = \frac{1}{m-1}\sum_{i=1}^{m}(x_{ij}-\mu_j)^2
$$

- Always $\ge 0$. It is zero only if every value is identical.
- Units are the feature's units **squared**. The **standard deviation** $\sigma_j=\sqrt{\operatorname{var}(x_j)}$ brings it back to the original units.

### Why $m-1$ and not $m$?

- **Population(${\sigma^2}$)** variance divides by $m$. Use it when you have every member of the population, or when $\mu$ is the *known true* mean.
- **Sample(${S^2}$)** variance divides by $m-1$. Use it when $\mu$ was estimated from the same data. Data sit closer to their own sample mean than to the true mean, so dividing by $m$ underestimates the spread. Dividing by $m-1$ helps to correct that bias.

For large $m$ the difference is negligible.

## Covariance

Covariance measures how two features vary **together**:

$$
\operatorname{cov}(x_j, x_k) = \frac{1}{m-1}\sum_{i=1}^{m}(x_{ij}-\mu_j)(x_{ik}-\mu_k)
$$

Look at each term: it is positive when both features are on the *same* side of their means, and negative when they are on *opposite* sides.

- **Positive:** when one is above its mean, the other tends to be too.
- **Negative:** when one is above its mean, the other tends to be below.
- **Near zero:** no *linear* relationship.

### Variance is covariance of a variable with itself

Plug the **same variable** into both spots of the covariance formula. For a variable $x$:

$$
\operatorname{cov}(x,x)
= \frac{1}{m-1}\sum_{i=1}^{m}(x_i-\mu_x)(x_i-\mu_x)
= \frac{1}{m-1}\sum_{i=1}^{m}(x_i-\mu_x)^2
= \operatorname{var}(x)
$$

The two factors in each term become identical, so the product is a square, and the result
is exactly the variance formula. Variance is the special case of
covariance where both variables are the same. The same check works with the example
tables below: in the covariance table, replacing the $y$ column with a copy of the $x$ column makes every product a square.

### Covariance is symmetric

$$
\operatorname{cov}(x,y)=\frac{1}{m-1}\sum_i(x_i-\mu_x)(y_i-\mu_y)
=\frac{1}{m-1}\sum_i(y_i-\mu_y)(x_i-\mu_x)=\operatorname{cov}(y,x)
$$

Each term is a product of the same two numbers, and multiplication does not care about
order.

## Correlation

Correlation rescales covariance by the two standard deviations:

$$
\rho_{jk} = \frac{\operatorname{cov}(x_j, x_k)}{\sigma_j\,\sigma_k}
$$

- Always in $[-1, 1]$ and has no units.
- $\rho = 1$: perfect positive linear relationship. $\rho=-1$: perfect negative. $\rho = 0$: no linear relationship.
- It is the covariance of the **standardized** features (each centered and divided by its standard deviation).

| | Covariance | Correlation |
|---|---|---|
| Range | Unbounded | $[-1, 1]$ |
| Units | Product of the features' units | Unitless |
| Changes if you rescale a feature (cm to m)? | Yes | No |

> Correlation tells us how *tightly* two features follow a line. It does not tell us how steep the line is.

## The covariance matrix

### Step 1: every pairwise covariance in one table

Take three variables $x$, $y$, $z$. Covariance compares two variables at a time, so the
natural thing to do is compute it for **every pair**, including each variable with itself,
and arrange the results in a grid. Row = first variable, column = second variable:

$$
C =
\begin{bmatrix}
\operatorname{cov}(x,x) & \operatorname{cov}(x,y) & \operatorname{cov}(x,z)\\
\operatorname{cov}(y,x) & \operatorname{cov}(y,y) & \operatorname{cov}(y,z)\\
\operatorname{cov}(z,x) & \operatorname{cov}(z,y) & \operatorname{cov}(z,z)
\end{bmatrix}
$$

### Step 2: simplify the diagonal

The diagonal entries pair a variable with itself, and we showed above that
$\operatorname{cov}(x,x)=\operatorname{var}(x)$:

$$
C =
\begin{bmatrix}
\operatorname{var}(x) & \operatorname{cov}(x,y) & \operatorname{cov}(x,z)\\
\operatorname{cov}(y,x) & \operatorname{var}(y) & \operatorname{cov}(y,z)\\
\operatorname{cov}(z,x) & \operatorname{cov}(z,y) & \operatorname{var}(z)
\end{bmatrix}
$$

### Step 3: simplify the off-diagonal using symmetry

Since $\operatorname{cov}(y,x)=\operatorname{cov}(x,y)$, and likewise for the other pairs, the
entries below the diagonal mirror the ones above:

$$
C =
\begin{bmatrix}
\operatorname{var}(x) & \operatorname{cov}(x,y) & \operatorname{cov}(x,z)\\
\operatorname{cov}(x,y) & \operatorname{var}(y) & \operatorname{cov}(y,z)\\
\operatorname{cov}(x,z) & \operatorname{cov}(y,z) & \operatorname{var}(z)
\end{bmatrix}
$$

So the covariance matrix holds **variances on the diagonal** and **covariances off the
diagonal**, and it is symmetric. For $n$ variables there are $n$ variances and
$\tfrac{n(n-1)}{2}$ distinct covariances, $\tfrac{n(n+1)}{2}$ unique numbers in total.

### Step 4: the same matrix as one matrix product

Computing every pair one at a time is tedious. Stack the centered data into $X_c$, where
each column is one variable (a column per $x$, $y$, $z$), and the whole matrix is a single
product:

$$
C = \frac{1}{m-1}X_c^TX_c \in \mathbb{R}^{n\times n}
$$

For two features $x_1,x_2$ this reads
$C=\begin{bmatrix}\operatorname{var}(x_1)&\operatorname{cov}(x_1,x_2)\\\operatorname{cov}(x_2,x_1)&\operatorname{var}(x_2)\end{bmatrix}$.

The product also explains the diagonal. Entry $(j,k)$ of $X_c^TX_c$ is the dot product of
column $j$ with column $k$. On the diagonal, $j=k$, so a column is dotted with *itself*, which
is its sum of squares, which is the variance (after dividing by $m-1$). Same fact, seen
through the matrix product instead of the formula.

Why is $X_c^TX_c$ the right product? Entry $(j,k)$ of $X_c^TX_c$ is the dot product of column $j$ with column $k$ of $X_c$, which is exactly $\sum_i (x_{ij}-\mu_j)(x_{ik}-\mu_k)$.

### Reading the matrix

- **Diagonal** $C_{jj}$: variance of feature $j$.
- **Off-diagonal** $C_{jk}$: covariance of features $j$ and $k$.
- **Trace** $\operatorname{tr}(C)$: the **total variance** of the dataset.

### Properties

- **Symmetric:** $C = C^T$.
- **Positive semi-definite:** $v^TCv \ge 0$ for every vector $v$, so all eigenvalues are $\ge 0$.
- Because it is symmetric, its eigenvalues are real and its eigenvectors can be chosen orthogonal. This is why `np.linalg.eigh` is the right solver for it.

### The correlation matrix

Divide each entry by the two relevant standard deviations:

$$
R_{jk} = \frac{C_{jk}}{\sqrt{C_{jj}C_{kk}}}
$$

It has 1s on the diagonal and is the covariance matrix of the standardized data.

## Worked examples by hand
*The below examples come from a video on Covarince by The OrganicChemistryTutor over on YouTube.*

### Positive covariance

$x=\langle 2,4,6,8,10\rangle$, $y=\langle 3,7,10,14,17\rangle$, $N=5$. Means: $\bar x = 6$, $\bar y = 10.2$.

| $x_i$ | $y_i$ | $x_i-\bar x$ | $y_i-\bar y$ | $(x_i-\bar x)(y_i-\bar y)$ |
|---|---|---|---|---|
| 2 | 3 | −4 | −7.2 | 28.8 |
| 4 | 7 | −2 | −3.2 | 6.4 |
| 6 | 10 | 0 | −0.2 | 0 |
| 8 | 14 | 2 | 3.8 | 7.6 |
| 10 | 17 | 4 | 6.8 | 27.2 |
| | | | **Sum** | **70.0** |

$$\operatorname{cov}(x,y)=\frac{70}{N-1}=\frac{70}{4}=17.5$$

Positive: as $x$ increases, $y$ increases.

### Negative covariance

$x=\langle 3,6,9,12,15\rangle$, $y=\langle 20,17,13,9,4\rangle$, $N=5$. Means: $\bar x = 9$, $\bar y = 12.6$.

| $x_i$ | $y_i$ | $x_i-\bar x$ | $y_i-\bar y$ | $(x_i-\bar x)(y_i-\bar y)$ |
|---|---|---|---|---|
| 3 | 20 | −6 | 7.4 | −44.4 |
| 6 | 17 | −3 | 4.4 | −13.2 |
| 9 | 13 | 0 | 0.4 | 0 |
| 12 | 9 | 3 | −3.6 | −10.8 |
| 15 | 4 | 6 | −8.6 | −51.6 |
| | | | **Sum** | **−120** |

$$\operatorname{cov}(x,y)=\frac{-120}{4}=-30$$

Negative: as $x$ increases, $y$ decreases.

> **Note:** a covariance of exactly 0 means no *linear* relationship, not no relationship.
> For example, $y=x^2$ over values symmetric around 0 has covariance 0 but is perfectly
> dependent.

## Common mistakes

- **Forgetting to center.** $X^TX$ on uncentered data is a second-moment matrix, not covariance.
- **Transposing the wrong way.** $X_c^TX_c$ gives $n\times n$ (features by features). $X_cX_c^T$ gives $m\times m$ (observations by observations).
- **Reading covariance magnitude as strength.** Use correlation for that.

## Implementation

- `covariance.py` (planned): `feature_means`, `center`, `covariance_matrix`, `correlation_matrix`