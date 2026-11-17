# Principal Component Analysis (PCA)

PCA is a linear algebra method for finding the directions along which data varies the most.
It rests on one idea: **the eigenvectors of the covariance matrix are the directions of
maximum variance, and the eigenvalues are the variances along them.**

Related:
- `../covariance/README.md`: variance, covariance, correlation, and the covariance matrix
- `../eigen/README.md`: eigenvalues and eigenvectors, $Av=\lambda v$

## The problem PCA solves

Features are often correlated, so the data does not fill its full space. It lies near a
lower-dimensional line or plane. Describing each point with all $n$ original coordinates is
then redundant.

PCA finds a new set of perpendicular axes, **ordered by how much the data varies along
each**. Keeping only the first $p$ axes compresses the data while preserving as much of its
variation as possible.

## What a principal component is

A **principal component** is a *direction* in feature space: a unit vector $v_k$.

- **PC1** is the direction along which the data has the largest variance.
- **PC2** is the direction of largest remaining variance that is **orthogonal** to PC1.
- **PC3** is the largest remaining, orthogonal to both PC1 and PC2. And so on, up to $n$ components.

A principal component is not a point in the data and not a score. It is an axis. The
distinction matters because four different things get called "the component":

| Term | Symbol | What it is | Shape |
|---|---|---|---|
| Principal component (direction) | $v_k$ | Unit vector; one column of $V$ | $(n,)$ |
| Eigenvalue | $\lambda_k$ | Variance of the data along $v_k$ | scalar |
| Score | $z_k = X_cv_k$ | Each observation's coordinate along $v_k$ | $(m,)$ |
| Explained variance ratio | $\lambda_k/\sum_j\lambda_j$ | Share of total variance along $v_k$ | scalar |

Each component is a recipe: a weighted combination of the original features. In a
2D example, $v_1=(0.816,\,0.578)$ means PC1 is $0.816\cdot x_1 + 0.578\cdot x_2$, a blend of
mostly $x_1$ with some $x_2$. A large weight means that feature contributes strongly to the
direction.

## How eigenvalues and eigenvectors connect to variance

The idea: *the covariance matrix describes how the data spreads. Its eigenvectors
are the directions of that spread, and its eigenvalues say how much spread there is
in each direction.*

### 1. Center the data and build the covariance matrix

Subtract the mean from every feature, then compute

$$
C = \frac{1}{m-1}X_c^TX_c
$$

$C$ is an $n \times n$ matrix. The diagonal holds each feature's variance, and the
off-diagonal entries show which features rise and fall together.

### 2. Find the eigenvectors and eigenvalues of C

An eigenvector $v$ of $C$ and its eigenvalue $\lambda$ satisfy

$$
Cv = \lambda v
$$

In words: when $C$ acts on $v$, the direction doesn't change. It only gets stretched
by $\lambda$. So eigenvectors are the "stable directions" of $C$.

- **Eigenvector** $v$: a direction in feature space (a principal component)
- **Eigenvalue** $\lambda$: how much the data varies along that direction

### 3. Why the eigenvalue is the variance

Project the centered data onto a unit vector $v$ to get scores $z = X_cv$. The
variance of those scores is

$$
\operatorname{Var}(z) = v^TCv
$$

If $v$ is an eigenvector, then $Cv = \lambda v$, so

$$
v^TCv = v^T(\lambda v) = \lambda\,(v^Tv) = \lambda
$$

because $v^Tv = 1$ for a unit vector. **The variance along an eigenvector is its
eigenvalue.**

### 4. Rank them!

Sort the eigenpairs by eigenvalue, largest first.

- **PC1**: eigenvector with the largest eigenvalue (most spread)
- **PC2**: next largest, perpendicular to PC1
- and so on

As we saw in the Covariance module, $C$ is symmetric. The eigenvectors of a symmetric matrix with distinct eigenvalues are
automatically orthogonal. So the PCA axes come out perpendicular without extra work. Their
scores are also **uncorrelated**: in the PC basis, the covariance matrix is diagonal.

$$
V^TCV = \Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_n)
$$

PCA is, in effect, a rotation of the axes into the eigenbasis of $C$, where all the
off-diagonal covariances vanish.

### Quick check: total variance is kept

The eigenvalues add up to the total variance, which is the trace of $C$:

$$
\sum_k \lambda_k = \operatorname{tr}(C)
$$

PCA only rotates the axes. It changes how the variance is split up, not how much
there is. That is why $\lambda_k / \sum\lambda_j$ (the explained variance ratio)
makes sense.

## The PCA algorithm

**Setup.** $X$ is your data with $m$ rows (observations) and $n$ columns (features).

The goal: find a few directions that capture most of the spread in the data, then
describe every observation using only those directions.

There are seven steps. The first four *find* the directions. The last three *use* them.

### Step 1: Center the data

Compute the mean of each column, then subtract it from every row.

$$
\mu = \text{column means of } X, \qquad X_c = X - \mathbf{1}\mu^T
$$

- Shapes: $\mu$ is $(n,)$, $X_c$ is $(m, n)$
- **Why:** variance measures spread *around the mean*. If you skip centering, the
  biggest "direction" PCA finds is just the line from the origin to the data's
  center, which says nothing about how the data varies.
- $\mu$ gets used again in Step 7. So make sure to keep it!

### Step 2: Compute the covariance matrix

$$
C = \frac{1}{m-1}X_c^TX_c
$$

- Shapes: $X_c^T$ is $(n, m)$ times $X_c$ is $(m, n)$, so $C$ is $(n, n)$
- $C$ is one number for every pair of features. The diagonal is each feature's
  variance, and the off-diagonal says how two features move together.
- The $m-1$ is the sample correction. The larger the sample, the less this matters.

### Step 3: Get the eigenvalues and eigenvectors of C

This is the step where the directions come from. Everything before it prepared
the covariance matrix, and everything after it uses the directions found here.

**What we're solving.** Find every pair $(\lambda, v)$ such that

$$
Cv = \lambda v
$$

Most vectors get rotated and stretched when you multiply them by $C$. An
eigenvector is a special vector that only gets *stretched* (by $\lambda$) and
stays on its own line. A symmetric $n \times n$ matrix has $n$ of these, and
they are perpendicular to each other.

**What it looks like by hand (2×2).** Rearranging gives $(C - \lambda I)v = 0$.
A nonzero $v$ only exists when $\det(C - \lambda I) = 0$, which for a 2×2 matrix
is a quadratic in $\lambda$:

$$
\lambda^2 - \operatorname{tr}(C)\,\lambda + \det(C) = 0
$$

By hand this works for 2×2. For 256×256 the polynomial approach is hopeless,
so we hand the matrix to a numerical routine.

**What `eigh` does.**

```python
values, vectors = np.linalg.eigh(C)
```

| | Shape | What it holds |
|---|---|---|
| `C` (input) | $(n, n)$ | symmetric covariance matrix |
| `values` | $(n,)$ | the eigenvalues $\lambda_k$ |
| `vectors` | $(n, n)$ | eigenvectors, **one per column** |

The eigenvector that goes with `values[k]` is `vectors[:, k]`, a column, not
`vectors[k]`, which is a row.

### Step 4: Sort largest first

```python
order = np.argsort(values)[::-1]
values, vectors = values[order], vectors[:, order]
```

- After sorting, column 0 of `vectors` is PC1, column 1 is PC2, and so on
- Sorting reorders the eigenvalues and the columns of `vectors` with the same
  `order`, so each eigenvalue stays attached to its own eigenvector. If you sort
  only one of them, you break the pairing and everything after this step is wrong.

**Steps 1 to 4 are what is called 'Fitting'**, this is when we apply the PCA algorithm to identify the principle components that capture the most variance in the data. It returns `mean`, `eigenvalues`, and
`eigenvectors`, and everything else is built from those three things.

### Step 5: Choose p and keep the first p eigenvectors

$$
V_p = \text{first } p \text{ columns of } V
$$

- Shape: $(n, p)$
- $p$ is *your choice*. It's how many directions you keep. In some cases you may want to test multiple $p$ sizes, for example:
  $$
  p \in \{10, 50, 100, 150, 200\}
  $$
- Choosing $p$ well has its own section. For now, a small $p$ value means more compression
  and more loss.

### Step 6: Project the scores (compress)

$$
Z = X_c V_p
$$

- Shapes: $(m, n)$ times $(n, p)$ gives $(m, p)$
- This is the compression step. Each observation went from $n$ numbers to $p$.
- Each entry of $Z$ is a **score**: how far that observation sits along one
  direction. Row $i$ of $Z$ is observation $i$ written in the new axes.
- Note that we must multiply the *centered* data, not the original $X$.
    - The directions $v_k$ were found from the *centered* data, so their axis run through the mean, not through the source.

### Step 7: Reconstruct

$$
\hat X = Z V_p^T + \mathbf{1}\mu^T
$$

- Shapes: $(m, p)$ times $(p, n)$ gives $(m, n)$, the original shape again
- **Why it works.** Any centered point is a weighted sum of the directions:

$$
x_c = z_1 v_1 + z_2 v_2 + \cdots + z_n v_n
$$

  The scores $z_k$ are the weights. Reconstruction keeps the first $p$ terms and
  drops the rest, so $\hat X$ is an *approximation*, not the original.
- **Why add $\mu$ back.** Step 1 removed it, so the reconstruction lives around zero
  until you put it back.

#### Why reconstruct the data? We already had the real values.

The purpose of reconstructing here is to gauge how well the first $p$ principal
components, $v_1, \dots, v_p$, represent the original data.

**The problem with $Z$ alone.** After projecting, each observation is just $p$
scores. You can't tell from those numbers whether anything important was lost,
and you can't compare them against $X$ because they're in different spaces
($m \times p$ versus $m \times n$). Reconstruction maps the scores back into the
original feature space, so $\hat X$ and $X$ have the same shape and can be
compared entry by entry.

**What does the comparison tell us?**

- **A number.** The MSE between $X$ and $\hat X$ measures how far the
  approximation is from the truth. It equals the variance of the components we
  dropped, so it also confirms the eigenvalue math from the Information Loss
  section.

**The reconstruction is not the real data.** It's what the data would look like if
only $p$ directions existed. Everything along the dropped directions is gone, so
$\hat X$ is smoother than $X$. It only matches $X$ exactly when
$p = n$.

**Where it's used for real, not just for comparison.**

- **Compression.** You store $Z$, $V_p$ and $\mu$ instead of $X$. That is
  $mp + np + n$ numbers instead of $mn$. When you need the data back, you
  reconstruct it. In that situation you no longer have the original to compare
  against, so $\hat X$ is all you've got.
- **Denoising.** The dropped directions often hold mostly noise, so $\hat X$ can be
  cleaner than $X$.

**When you don't need it.** If you're feeding $Z$ straight into something else,
like a classifier or a 2D plot, you never go back to the original space, so
there's nothing to reconstruct. 

### Summary

| Step | What | Result | Shape |
|---|---|---|---|
| 1 | Center | $X_c$ | $(m, n)$ |
| 2 | Covariance | $C$ | $(n, n)$ |
| 3 | Eigendecompose | $\lambda$, $V$ | $(n,)$, $(n, n)$ |
| 4 | Sort | same, reordered | same |
| 5 | Keep $p$ | $V_p$ | $(n, p)$ |
| 6 | Project | $Z$ | $(m, p)$ |
| 7 | Reconstruct | $\hat X$ | $(m, n)$ |

## Information loss

Reducing from $n$ dimensions to $p$ throws information away. This section shows
exactly how much, and the answer comes straight from the eigenvalues.

### What gets lost

Recall from Step 7 that every centered point is a sum of its directions:

$$
x_c = z_1 v_1 + z_2 v_2 + \cdots + z_n v_n
$$

Reconstructing with $p$ components keeps the first $p$ terms. So the **error for
each point is exactly the terms you dropped**:

$$
x_c - \hat x_c = z_{p+1} v_{p+1} + \cdots + z_n v_n
$$

Nothing else is lost. The kept terms are reproduced perfectly.

## Choosing p

There is no single correct $p$. It's a trade-off between compression and
fidelity, and the right balance depends on what you're doing with the result. The
eigenvalues give you three tools for making the right call for your usecase.

### 1. Explained variance ratio (EVR)

The share of the total variance carried by one component:

$$
\text{EVR}_k = \frac{\lambda_k}{\sum_j \lambda_j}
$$

This works because total variance is preserved ($\sum\lambda_j = \operatorname{tr}(C)$),
so the ratios add up to 1.

### 2. Cumulative explained variance

The share carried by the first $p$ components together:

$$
\text{cumulative}(p) = \frac{\sum_{k\le p}\lambda_k}{\sum_j \lambda_j}
$$

Pick a target (90% and 95% are common rules of thumb) and keep the smallest $p$
that reaches it.

```python
ratios = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(ratios)
p = np.searchsorted(cumulative, 0.95) + 1     # smallest p reaching 95%
```

The `+ 1` is there because index 0 means one component.

**Connection to error:** variance *kept* is cumulative$(p)$, and variance *lost*
is $1 - \text{cumulative}(p)$.

### 3. Scree plot

Plot $\lambda_k$ against $k$. Typically the curve drops steeply, then flattens
into a long tail. The bend is the "elbow".

- Components **before** the elbow each add a lot
- Components **after** the elbow each add very little

Choosing $p$ near the elbow is a judgment call, not a formula, which is why it's
usually paired with a cumulative target.

### Practical notes

- **Bigger $p$ is not always good.** More components means more storage and less
  compression, and keeping the small ones can keep noise.
- **Choose by purpose.** 
  - *Visualization:* $p = 2$ or $3$.
  - *Compression:* use a **cumulative variance** target, since variance lost is
    exactly the reconstruction error.
  - *Classification:* a variance target is a good first guess, but variance is not
    the same as usefulness(*Bigger $P$ is not always good!*), so confirm by testing accuracy at several $p$ values.
- **Tiny negative eigenvalues** (for exmaple something like $-10^{-16}$) can appear in the tail of the
  data. They are rounding errors for values that are really 0.