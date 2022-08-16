---
title: "Modal Decomposition"
mathjax: true
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

We start by defining an _inner product_ in the functional space. Given an alphabet $\mathcal X$, the space of all real-valued functions, 

$$
\mathcal {F_X} = \{f: \mathcal X \mapsto \mathbb R \},
$$

can be viewed as a vector space. Here, we need to fix a distribution $R_\mathsf x$ on $\mathcal X$, which we call the _reference distribution_. Based on that we can define the inner product: for any $f_1, f_2 \in \mathcal F_\mathcal X$, 

$$
\langle f_1, f_2\rangle \stackrel{\Delta}{=} \mathbb E_{\mathsf x \sim R_X}[f_1(\mathsf x) \cdot f_2(\mathsf x)]
$$

---
**Note:**
In almost all cases we can without loss of generality restrict functions to have zero mean w.r.t. $R_\mathsf x$. Thus, the inner product is really the covariance of $f_1(\mathsf x)$ and $f_2(\mathsf x)$. Furthermore, in this page we would not change the reference distribution once chosen, so we could use the above notation for inner products. Otherwise we could put a subscript to indicate the reference, like $\langle f_1, f_2\rangle_{R_\mathsf x}$. 
---

We can similarly define the inner product on the space of functions on a different alphabet $\mathcal Y$, with respect to a reference distribution $R_\mathsf y$. 

Now we are ready to address the joint distributions $P_{\mathsf {xy}}$ on $\mathcal {X\times Y}$. Again we need to choose a reference distribution $R_\mathsf {xy}$. For the purpose of this page, we use the product distribution $R_{\mathsf {xy}} = R_\mathsf x\cdot R_\mathsf y$, and take the resulting definition of inner product of functions in $\mathcal F_{\mathcal X\times \mathcal Y}$. 

A particular function of interest in $\mathcal {F_{X\times Y}}$ is the log likelihood ratio

$$ 
\mathrm{LLR}(x,y) \stackrel{\Delta}{=}\log \frac{P_{\mathsf {xy}}(x,y)}{P_{\mathsf x}(x) \cdot P_{\mathsf y}(y)}, \quad x\in \mathcal X, y \in \mathcal Y
$$

where $P_\mathsf x$ and $P_\mathsf y$ are the $\mathsf x$ and $\mathsf y$ marginal distributions of $P_\mathsf {xy}$. It is clear from the definition that $\mathrm{LLR}(x,y) = 0, \forall x$ if and only if the two random variables $\mathsf{x, y}$ are independent; and in general this function gives a complete description of how the two are dependent to each other. Consequently, the LLR function, or in some equivalent or reduced forms, is the target of almost all learning problems. The main difficulty in practice is that the alphabets $\mathcal {X, Y}$ are often very big, causing the LLR function to be very high dimensional, which makes these learning tasks difficult. 

Here, we first need to make a technical assumption. We assume that the joint distribution $P_{\mathsf {xy}}$ satisfies there exists a possibly infinite sequence of pairs of functions $(f_i, g_i), f_i \in \mathcal {F_X}, g_i \in \mathcal {F_Y}, i=1, 2, \ldots$, such that 

$$
\lim_{n\to \infty} \left\Vert \mathrm{LLR} - \sum_{i=1}^n f_i \otimes g_i \right\Vert^2 = \lim_{n\to \infty} \mathbb E_{\mathsf {x,y} \sim R_\mathsf xR_\mathsf y}\left[ \left(\mathrm{LLR}(\mathsf {x, y}) - \sum_{i=1}^n f_i(\mathsf x) g_i(\mathsf y) \right)^2\right]  = 0
$$


