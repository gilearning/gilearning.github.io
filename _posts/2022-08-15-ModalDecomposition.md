---
title: "Modal Decomposition"
layout: post
---
## Modal Decomposition of Statistical Dependence


> ### The key points
> 
>_Statistical dependence_ is the reason that we can guess the value of one random variable based on the observation of another. The is the basis of most inference problems like decision making, estimation, prediction, classification, etc. 
>
>It is, however, a somewhat ill-posed question to ask "how much does a random variable $\mathsf x$ depend on another random variable $\mathsf y$". 
>It turns out that in general statistical dependence should be understood and quantified as a high dimensional relation: two random variables are dependent through a number of **_orthogonal modes_**; and each mode can have a different "strength". 
>
>The goal of this page is to define these modes mathematically, explain why they are important in practice, and show by examples that many statistical concepts and learning algorithms are directly related to this modal decomposition idea. With that we will also build the mathematical foundation and notations for the more advanced processing using modal decomposition in the later pages. 

### Inner Product

We start by defining an _inner product_ in the functional space. Given an alphabet $\mathcal X$, the space of all real-valued functions, 

$$
\mathcal {F_X} = \{f: \mathcal X \mapsto \mathbb R \},
$$

can be viewed as a vector space. Here, we need to fix a distribution $R_\mathsf x$ on $\mathcal X$, which we call the _reference distribution_. Based on that we can define the inner product: for any $f_1, f_2 \in \mathcal F_\mathcal X$, 

$$
\langle f_1, f_2\rangle \stackrel{\Delta}{=} \mathbb E_{\mathsf x \sim R_X}[f_1(\mathsf x) \cdot f_2(\mathsf x)]
$$


>**Note:**
>In almost all cases we can without loss of generality restrict functions to have zero mean w.r.t. $R_\mathsf x$. Thus, the inner product is really the covariance of $f_1(\mathsf x)$ and $f_2(\mathsf x)$. Furthermore, in this page we would not change the reference distribution once chosen, so we could use the above notation for inner products. Otherwise we could put a subscript to indicate the reference, like $\langle f_1, f_2\rangle_{R_\mathsf x}$. 


We can similarly define the inner product on the space of functions on a different alphabet $\mathcal Y$, with respect to a reference distribution $R_\mathsf y$. 

### The LLR Function

Now we are ready to address the joint distributions $P_{\mathsf {xy}}$ on $\mathcal {X\times Y}$. Again we need to choose a reference distribution $R_\mathsf {xy}$. For the purpose of this page, we use the product distribution $R_{\mathsf {xy}} = R_\mathsf x\cdot R_\mathsf y$, and take the resulting definition of inner product of functions in $\mathcal F_{\mathcal X\times \mathcal Y}$. 

A particular function of interest in $\mathcal {F_{X\times Y}}$ is the log likelihood ratio

$$ 
\mathrm{LLR}(x,y) \stackrel{\Delta}{=}\log \frac{P_{\mathsf {xy}}(x,y)}{P_{\mathsf x}(x) \cdot P_{\mathsf y}(y)}, \quad x\in \mathcal X, y \in \mathcal Y
$$

where $P_\mathsf x$ and $P_\mathsf y$ are the $\mathsf x$ and $\mathsf y$ marginal distributions of $P_\mathsf {xy}$. It is clear from the definition that $\mathrm{LLR}(x,y) = 0, \forall x$ if and only if the two random variables $\mathsf{x, y}$ are independent; and in general this function gives a complete description of how the two are dependent to each other. Consequently, the LLR function, or in some equivalent or reduced forms, is the target of almost all learning problems. The main difficulty in practice is that the alphabets $\mathcal {X, Y}$ are often very big, causing the LLR function to be very high dimensional, which makes these learning tasks difficult. 

Here, we need to make a technical assumption. For a pair of functions $f \in \mathcal {F_X}$ and $g \in \mathcal {F_Y}$, we denote $f\otimes g \in \mathcal {F_{X\times Y}}$ as the "tensor product function" or simply the product function, with $f\otimes g(x,y) \stackrel{\Delta}{=} f(x) g(y), \forall x, y$. Now we assume that the joint distribution $P_{\mathsf {xy}}$ satisfies there exists a possibly infinite sequence of pairs of functions $(f_i, g_i), f_i \in \mathcal {F_X}, g_i \in \mathcal {F_Y}, i=1, 2, \ldots$, such that 

$$
\lim_{n\to \infty} \left\Vert \mathrm{LLR} - \sum_{i=1}^n f_i \otimes g_i \right\Vert^2 = \lim_{n\to \infty} \mathbb E_{\mathsf {x,y} \sim R_\mathsf xR_\mathsf y}\left[ \left(\mathrm{LLR}(\mathsf {x, y}) - \sum_{i=1}^n f_i(\mathsf x) g_i(\mathsf y) \right)^2\right]  = 0
$$

>**Note:**
>In words, this assumption says that the LLR function can be approached, in L2 sense, by the sum of a countable collection of product functions, with L2 defined w.r.t. the given reference distribution. This assumption is always true for the cases that both $\mathcal X$ and $\mathcal Y$ are discrete alphabets. For more general cases, the assumption of a countable basis in L2 sense is a commonly used assumption, which is not restrictive at all in most practical applications, and convenient for us to rule out some of the "unpleasant" distributions. 

### A Single Mode

Why we are so intersted in such product functions? In short, it represents a very simple kind of dependence. Imagine a joint distribution $P_{\mathsf {xy}}$ whose LLR function can be written as 

$$
\log \frac{P_{\mathsf {xy}}(x,y)}{P_\mathsf x(x) P_\mathsf y(y)} = f(x) \cdot g(y), \qquad \forall x, y.
$$



This can be rewritten as $P_{\mathsf {y|x}}(y|x) = P_\mathsf y (y) \cdot \exp(f(x)\cdot g(y)), \forall x, y$. That is, the conditional distribution is on a 1-D exponential family with $g(\mathsf y)$ as the natural statistic. To make inference of $\mathsf y$, we only need to know the value $f(\mathsf x)$, which is a sufficient statistic. And in fact the only thing we can infer about $\mathsf y$ is the value of $g(\mathsf y)$. In general, we could extrapolate from this observation to state that if the LLR function is the sum of a limited number of product functions, then that correspondingly limits the scope of inference tasks we can hope to solve, while allowing us to only look at a limited set of statistics, or **_features_**, of the data. 




This is a good time to give the definition of features and modes. 

---
**Definition: Feature Functions**
A feature function of data $x \in \mathcal X$ is $f \in \mathcal {F_X}$, with 

$$
\mathbb E_{\mathsf x \sim R_\mathsf x} [f(\mathsf x)] = 0, \qquad \mathbb E_{\mathsf x \sim R_\mathsf x}[ f^2(\mathsf x)] = 1
$$

---


Since these are the functions that we would like to evaluate with data, fixed shifting and scaling do not make any difference, so we included in the definition that feature functions must have zero mean and unit variance w.r.t. the given reference distribution. Because we normalize all feature functions from now on, when we write a product function, we need to explicitly write out the scaling factor. That is, instead of $f\otimes g$, we need to write $\sigma f\otimes g$, with $\sigma \geq 0$. We call this triple, $(\sigma, f, g)$, a single **_mode_**. That is, a mode consists of a strength $\sigma$, and a pair of feature functions in $\mathcal {F_X}$ and $\mathcal {F_Y}$. 

For a given data sample $x$, we think of the function value $f(x)$ a feature, which captures some partial information carried by the raw data, since in general we may not be able to recover the value of $x$ from the feature value $f(x)$. When we observe a sequence samples, $x_1, x_2, \ldots, x_n$, the corresponding feature is the empirical average $\frac{1}{n} (f(x_1) + \ldots + f(x_n))$. 

### Modal Decomposition

Obviously, for a given LLR function, there are many ways to write it into sum of modes. We hope to have as few modes as possible. In some cases, we might even wish to compromise the precision of the sum and try to have a reasonable approximation of the given LLR with a sum of even fewer modes. That is, for a given finite $k$, we would like to solve the problem 

$$
\min_{ (\sigma_i, f_i, g_i), i=1, \ldots, k} \, \left \Vert \mathrm{LLR} - \sum_{i=1}^k \sigma_i f_i\otimes g_i\right\Vert^2
$$


This optimization is in fact a well-studied one. For the case with finite alphabets, the target LLR function can be thought as a $|\mathcal X| \times |\mathcal Y|$ matrix, with the $(x,y)$ entry being the function value $\mathcal {LLR}(x,y)$; and the above optimization problem is solved by finding the singular value decomposition (SVD) of this matrix. The result is a decomposition is a diagonlization, turning LLR matrix into the sum of orthogonal rank-1 matrices, each corresponds to one mode in our definition. These optimal choice of modes must be orthogonal to each other as a result to avoiding repetition between modes. We will state here without proof that the same can be done in the functional space. With that we now give the definition of modal decomposition. 


---

**Definition: Modal Decomposition $\zeta$**

For a pair of spaces $\mathcal {F_X}$, $\mathcal {F_Y}$, with corresponding reference distributions $R_\mathsf x, R_\mathsf y$, resp., the **modal decomposition operation** is a map $\zeta$ that maps a joint distribution $P_{\mathsf {xy}}$, satisfying the technical assumption above, to a sequence of modes $(\sigma_i, f_i, g_i), i = 1, 2, \ldots$, with 

* $\sigma_i > 0, \forall i$; 
* $\sigma_1 \geq \sigma_2 \geq \ldots$ in descending order; 
* $f_i \in \mathcal {F_X}, g_i \in \mathcal {F_Y}$ are valid feature functions (zero meaan unit variance w.r.t the corresponding references)
* $\langle f_i, f_j\rangle = \langle g_i, g_j \rangle = \delta_{ij}$. 

These modes satisfy that 

$$
\begin{align}
a&=b\\
c&=d
\end{align}
$$

---
