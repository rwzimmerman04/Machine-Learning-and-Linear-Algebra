# Support Vector Machine (SVM)

> Study notes. Prerequisite: [PERCEPTRON.md](../perceptron/PERCEPTRON.md) (hyperplanes, the $y\,w^\top x$ correctness test, margin $\gamma$, linear separability).

---

## 1. Setup: from perceptron to SVM

Same ingredients as the [perceptron](../perceptron/PERCEPTRON.md), now with an explicit bias $b$:

| Item | Definition |
|---|---|
| Data | $x \in \mathbb{R}^d$, label $y \in \{-1, +1\}$ |
| Score | $w^\top x + b$ |
| Prediction | $h(x) = \operatorname{sign}(w^\top x + b)$ |
| Output | $h(x) > 0 \Rightarrow +1$, $\ h(x) < 0 \Rightarrow -1$ |

Correctness test (same idea as the perceptron's $y\,w^\top x$):

$$y\,(w^\top x + b) = y \cdot (\text{raw score})$$

- $> 0$: **correct**
- $\le 0$: **incorrect**

---

## 2. The key question: what is the best hyperplane?

When the data is linearly separable, there are **infinitely many** separating hyperplanes. The perceptron returns *any* one of them. Some are clearly worse than others.

**SVM's answer:** pick the hyperplane that **maximizes the distance from the closest data point on either side.**

---

## 3. Deriving the margin

Goal: the distance from a point $x_1$ to the hyperplane $w^\top x + b = 0$.

**Setup.**
- $x_p$ is the point on the hyperplane closest to $x_1$ (its projection).
- $d$ is the vector from $x_p$ to $x_1$. It is perpendicular to the hyperplane, so it is parallel to $w$:

$$d = \alpha w, \qquad x_p = x_1 - d$$

**Use the fact that $x_p$ is on the hyperplane:** $w^\top x_p + b = 0$.

$$w^\top (x_1 - d) + b = 0$$

$$w^\top (x_1 - \alpha w) + b = 0$$

$$w^\top x_1 + b = \alpha\, w^\top w$$

$$\alpha = \frac{w^\top x_1 + b}{w^\top w}$$

**Length of $d$:**

$$\|d\| = \|\alpha w\| = |\alpha|\,\|w\| = \frac{|w^\top x_1 + b|}{\|w\|^2}\,\|w\|$$

$$\boxed{\|d\| = \frac{|w^\top x_1 + b|}{\|w\|}}$$

(using $w^\top w = \|w\|^2$).

**Margin of the whole hyperplane** = distance of the *closest* point:

$$\gamma(w, b) = \min_{(x,y)\in D} \frac{|w^\top x + b|}{\|w\|}$$

> The margin is the minimum distance from any point to the hyperplane.

---

## 4. Scale invariance

By definition, the **hyperplane and the margin are scale invariant.** Scaling $w \to \alpha w$ and $b \to \alpha b$ gives:

$$\frac{|\alpha w^\top x + \alpha b|}{\|\alpha w\|} = \frac{|\alpha|\,|w^\top x + b|}{|\alpha|\,\|w\|} = \frac{|w^\top x + b|}{\|w\|}$$

Same hyperplane, same margin. We will **exploit this freedom** in section 6: we are allowed to choose the scale however we want.

---

## 5. The max margin classifier

We have defined $\gamma(w,b)$. Now **maximize the margin** to get a maximizing classifier:

$$\arg\max_{w,b}\ \gamma(w,b) \;=\; \arg\max_{w,b}\ \min_{(x,y)\in D} \frac{|w^\top x + b|}{\|w\|}$$

This is the problem we solve for the SVM.

---

## 6. Fixing the objective: constraint and scaling

### Step 1: add a correctness constraint

As written, the objective has a flaw: it only uses *distance* $|w^\top x + b|$, not whether points are on the *correct side*. Without a constraint, the maximum is achieved by a hyperplane placed **very far away** from all the data, which classifies badly.

Add the constraint:

$$\text{s.t.}\quad y\,(w^\top x + b) \ge 0 \quad \forall\, (x,y) \in D$$

("s.t." = "such that"). This forces every point onto its correct side.

```
 o  o             []  []
 o  o   | |        []  []              \
  o     | |        []                   \
        (good)                    (bad: big distance, useless)
```
Without the constraint, the "max margin" is a hyperplane far off to the side. With the constraint, the absolute value is no longer needed, since $y(w^\top x+b)$ is nonnegative for every point.

$$\arg\max_{w,b}\ \frac{1}{\|w\|}\ \min_{(x,y)\in D} |w^\top x + b| \quad \text{s.t.}\ \ y(w^\top x + b) \ge 0\ \ \forall (x,y)$$

### Step 2: choose the scale

Because the hyperplane is scale invariant, *"we can scale it however we want."* **Choose the scale so that:**

$$\min_{(x,y)\in D} |w^\top x + b| = 1$$

The closest point(s) now sit exactly at $w^\top x + b = \pm 1$.

```
        w^T x + b = -1      w^T x + b = 0      w^T x + b = +1
   []        :                    |                  :        o
   []        :                    |                  :      o
   []        :                    |                  :        o
```

Then the objective collapses to just $\frac{1}{\|w\|}$, and the constraint tightens:

- every point satisfies $|w^\top x + b| \ge 1$ (since the minimum is 1),
- and correctness requires the right sign,

so together:

$$y\,(w^\top x + b) \ge 1 \quad \forall\, (x,y) \in D$$

---

## 7. Final hard margin SVM

$$\arg\max_{w,b}\ \frac{1}{\|w\|} \quad \text{s.t.}\ \ y\,(w^\top x + b) \ge 1\ \ \forall (x,y)\in D$$

Maximizing $\frac{1}{\|w\|}$ is the same as **minimizing $\|w\|$**, which is the same as **minimizing $\|w\|^2$**:

$$\boxed{\ \arg\min_{w,b}\ \|w\|^2 \quad \text{s.t.}\ \ y\,(w^\top x + b) \ge 1\ \ \forall (x,y)\in D\ }$$

**Why $\|w\|^2$?** Minimizing the norm means minimizing a **convex** (bowl-shaped) function, which is smooth and easy to optimize. This is a **Quadratic Program** (quadratic objective, linear constraints).

*(Textbooks often write $\tfrac{1}{2}\|w\|^2$. The $\tfrac12$ just simplifies the derivative and does not change the answer.)*

**Geometry recap:** the margin on each side is $\frac{1}{\|w\|}$, so the total gap between the two margin lines is $\frac{2}{\|w\|}$.

---

## 8. Support vectors

**Support vectors** are the data points that lie *exactly* on the margin boundaries:

$$y\,(w^\top x + b) = 1$$

- They are called support vectors because **in SVM, the margin is defined by these vectors.** Move or remove any other point and the solution does not change.
- They are the **tightening points**: the constraints are **"tight"** (active, holding with equality). "Tight" is a term from optimization.

```
   o     \  :  |  :  /      []
 o   [o]  \ :  |  : /  [[]]     <- circled points on the dashed lines
   o  [o]    :  |  :    []         are the support vectors
              margin  margin
```

---

## 9. Soft margin SVM

**Remember:** the [perceptron](../perceptron/PERCEPTRON.md) only works when the data is linearly separable. The **hard margin SVM is the same way.** If no hyperplane separates the data, the constraints $y(w^\top x+b)\ge 1$ cannot all be satisfied.

But the developers of SVM did not give up. They asked: *can we design a relaxed approach rather than hard margin SVM?*

### The idea: slack variables

Allow each point to **violate** its constraint a little, by an amount $\xi_i$ ("xi", the **slack**):

$$y_i\,(w^\top x_i + b) \ge 1 - \xi_i, \qquad \xi_i \ge 0$$

"Relax it": $\xi_i = 0$ recovers the hard constraint.

### Soft margin objective

$$\arg\min_{w,b,\xi}\ \|w\|^2 + C\sum_i \xi_i \quad \text{s.t.}\ \ y_i(w^\top x_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0$$

Two competing goals:
- $\|w\|^2$ small $\Rightarrow$ **wide margin**
- $\sum \xi_i$ small $\Rightarrow$ **few/small violations**

(The notes sketch this with $C = 1$. The general form has a weight $C$, discussed below.)

### What $\xi_i$ means

| $\xi_i$ | Where the point is |
|---|---|
| $\xi_i = 0$ | Correct side, outside or on the margin |
| $0 < \xi_i < 1$ | Correct side, but **inside** the margin |
| $\xi_i = 1$ | Exactly on the hyperplane |
| $\xi_i > 1$ | **Wrong side** (misclassified) |

```
 correct side          wrong side         correct side
   (circles)                                (squares)
   o  o  o    |  [o]  |  []  [] [] 
   o   o      |   ..  |   []  []        circled "incorrect" SVs sit on the wrong side
              y(w^T x + b) = 1           or inside the margin
```

**Support vectors in soft margin:** now include points *on* the margin, *inside* the margin, and *on the wrong side* (the "incorrect" SVs).

---

## 10. Soft margin on XOR and the role of C

Try the soft margin SVM on the **XOR problem** (not linearly separable, so a hard margin SVM has no solution).

- Pick a vertical line as the hyperplane. Some points are on the wrong side, so they need slack.
- Example from the notes: a misclassified point with $\xi_i = 2$ satisfies
  $$y(w^\top x + b) \ge 1 - \xi_i = 1 - 2 = -1$$
  i.e. the constraint is relaxed down to $-1$.
- This is **one feasible solution with the soft margin SVM** (some points have $\xi_i = 0$, the unlucky ones have large $\xi_i$).

### C = "level of strictness" on the slack

| Cost $C$ on $\xi_i$ | Effect |
|---|---|
| **Low** $C$ | Slack is cheap, so violations are tolerated and a solution is possible |
| **High** $C$ | Slack is expensive, so behavior approaches hard margin; on data like XOR there is effectively "no solution" |

> $C$ is our "level of strictness" on the slack $\xi_i$.
>
> Clarification: mathematically, the soft margin problem always has *some* feasible answer (just make $\xi$ big). "No solution" for high $C$ means it behaves like the hard margin problem, which cannot satisfy XOR.

Note that a *linear* soft margin SVM still cannot truly *solve* XOR (it only tolerates the errors). Handling it properly takes a non-linear approach.
