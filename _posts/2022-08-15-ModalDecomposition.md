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
In almost all cases we can without loss of generality restrict functions to have zero mean w.r.t. $R_\mathsf x$. Thus, the inner product is really the covariance of $f_1(\mathsf x)$ and $f_2(\mathsf x)$.
---

We can similarly define the inner product on the space of functions on a different alphabet $\mathcal Y$, with respect to a reference distribution $R_\mathsf y$. 

Now we are ready to address the joint distributions $P_{\mathsf {xy}}$ on $\mathcal {X\times Y}$. Again we need to choose a reference distribution $R_\mathsf {xy}$. For the purpose of this page, we use the product distribution $R_{\mathsf {xy}} = R_\mathsf x\cdot R_\mathsf y$. 
