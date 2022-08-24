---
title: "Modal Decomposition"
layout: post
---

+ a
+ b
* TOC
{:toc}

> ### The key points
> 
>_Statistical dependence_ is the reason that we can guess the value of one random variable based on the observation of another. The is the basis of most inference problems like decision making, estimation, prediction, classification, etc. 
>
>It is, however, a somewhat ill-posed question to ask, "how much does a random variable $\mathsf x$ depend on another random variable $\mathsf y$". 
>It turns out that in general statistical dependence should be understood and quantified as a high dimensional relation: two random variables are dependent through a number of **_orthogonal modes_**; and each mode can have a different "strength". 
>
>The goal of this page is to define these modes mathematically, explain why they are important in practice, and show by examples that many statistical concepts and learning algorithms are directly related to this modal decomposition idea. With that we will also build the mathematical foundation and notations for the more advanced processing using modal decomposition in the later pages. 

## Modal Decomposition of Statistical Dependence
### Inner Product


We start by defining an _inner product_ in the functional space. Given an alphabet $\mathcal X$, the space of all real-valued functions, 

$$
\mathcal {F_X} = \{f: \mathcal X \to \mathbb R \},
$$

can be viewed as a vector space. Here, we need to fix a distribution $R_\mathsf x$ on $\mathcal X$, which we call the _reference distribution_. Based on that we can define the inner product: for any $f_1, f_2 \in \mathcal F_\mathcal X$, 

$$
\langle f_1, f_2\rangle \stackrel{\Delta}{=} \mathbb E_{\mathsf x \sim R_X}[f_1(\mathsf x) \cdot f_2(\mathsf x)]
$$


>**Note:**
>In almost all cases we can without loss of generality restrict functions to have zero mean w.r.t. $R_\mathsf x$. Thus, the inner product is really the covariance of $f_1(\mathsf x)$ and $f_2(\mathsf x)$. Furthermore, in this page we would not change the reference distribution once chosen, so we could use the above notation for inner products. Otherwise we could put a subscript to indicate the reference, like $\langle f_1, f_2\rangle_{R_\mathsf x}$. 


We can similarly define the inner product on the space of functions on a different alphabet $\mathcal Y$, with respect to a reference distribution $R_\mathsf y$. 

### The PMI Function

Now we are ready to address the joint distributions $P_{\mathsf {xy}}$ on $\mathcal {X\times Y}$. Again we need to choose a reference distribution $R_\mathsf {xy}$. For the purpose of this page, we use the product distribution $R_{\mathsf {xy}} = R_\mathsf x\cdot R_\mathsf y$, and take the resulting definition of inner product of functions in $\mathcal F_{\mathcal X\times \mathcal Y}$. 

A particular function of interest in $\mathcal {F_{X\times Y}}$ is the [Point-wise Mutual Information (PMI)](https://en.wikipedia.org/wiki/Pointwise_mutual_information)

$$ 
\mathrm{PMI}(x,y) \stackrel{\Delta}{=}\log \frac{P_{\mathsf {xy}}(x,y)}{P_{\mathsf x}(x) \cdot P_{\mathsf y}(y)}, \quad x\in \mathcal X, y \in \mathcal Y
$$

where $P_\mathsf x$ and $P_\mathsf y$ are the $\mathsf x$ and $\mathsf y$ marginal distributions of $P_\mathsf {xy}$. It is clear from the definition that $\mathrm{PMI}(x,y) = 0, \forall x$ if and only if the two random variables $\mathsf{x, y}$ are independent; and in general this function gives a complete description of how the two are dependent to each other. Consequently, the PMI function, or in some equivalent or reduced forms, is the target of almost all learning problems. The main difficulty in practice is that the alphabets $\mathcal {X, Y}$ are often very big, causing the PMI function to be very high dimensional, which makes these learning tasks difficult. 

Here, we need to make a technical assumption. For a pair of functions $f \in \mathcal {F_X}$ and $g \in \mathcal {F_Y}$, we denote $f\otimes g \in \mathcal {F_{X\times Y}}$ as the "tensor product function" or simply the product function, with $f\otimes g(x,y) \stackrel{\Delta}{=} f(x) g(y), \forall x, y$. Now we assume that the joint distribution $P_{\mathsf {xy}}$ satisfies there exists a possibly infinite sequence of pairs of functions $(f_i, g_i), f_i \in \mathcal {F_X}, g_i \in \mathcal {F_Y}, i=1, 2, \ldots$, such that 

$$
\lim_{n\to \infty} \left\Vert \mathrm{PMI} - \sum_{i=1}^n f_i \otimes g_i \right\Vert^2 = \lim_{n\to \infty} \mathbb E_{\mathsf {x,y} \sim R_\mathsf xR_\mathsf y}\left[ \left(\mathrm{PMI}(\mathsf {x, y}) - \sum_{i=1}^n f_i(\mathsf x) g_i(\mathsf y) \right)^2\right]  = 0
$$

>**Note:**
>In words, this assumption says that the PMI function can be approached, in L2 sense, by the sum of a countable collection of product functions, with L2 defined w.r.t. the given reference distribution. This assumption is always true for the cases that both $\mathcal X$ and $\mathcal Y$ are discrete alphabets. For more general cases, the assumption of a countable basis in L2 sense is a commonly used assumption, which is not restrictive at all in most practical applications, and convenient for us to rule out some of the "unpleasant" distributions. 

### A Single Mode

Why we are so interested in such product functions? In short, it represents a very simple kind of dependence. Imagine a joint distribution $P_{\mathsf {xy}}$ whose PMI function can be written as 

$$
\log \frac{P_{\mathsf {xy}}(x,y)}{P_\mathsf x(x) P_\mathsf y(y)} = f(x) \cdot g(y), \qquad \forall x, y.
$$


This can be rewritten as $P_{\mathsf {y\vert x}}(y\vert x) = P_\mathsf y (y) \cdot \exp(f(x)\cdot g(y)), \forall x, y$. 
That is, the conditional distribution is on a 1-D exponential family with $g(\mathsf y)$ as the natural statistic. 
To make inference of $\mathsf y$, we only need to know the value $f(\mathsf x)$, which is a sufficient statistic. 
In fact the only thing we can infer about $\mathsf y$ is the value of $g(\mathsf y)$. 
In general, we could extrapolate from this observation to state that if the PMI function is the sum of a limited number of product functions, then that correspondingly limits the scope of inference tasks we can hope to solve, while allowing us to only look at a limited set of statistics, or **_features_**, of the data. 

Here, to clarify the terminology, we refer to _feature functions_ of a random variable as real-valued functions on the alphabet, such as $f: \mathcal X \to \mathbb R$. Feature functions are often evaluated with the observed data samples, and the function values, which we refer to as _features_, are used for further inference and learning tasks, instead of the raw data. Thus, these features are indeed the "information carrying device". Since any known shifting and scaling do not change the information contents that these features carry, for convenience, we sometimes require a standard form, that the feature functions satisfying $\mathbb E[f(\mathsf x)] = 0$ and $\mathbb E[f^2(\mathsf x)]=1$, where both expectations are takend w.r.t. the reference distribution $R_\mathsf x$. 

When we write a product function as above in this standard form, we need to explicitly write out the scaling factor. That is, instead of $f\otimes g$, we need to write $\sigma f\otimes g$, with $\sigma \geq 0$. We call this triple, $(\sigma, f, g)$, a single **_mode_**. That is, a mode consists of a strength $\sigma$, and a pair of feature functions in $\mathcal {F_X}$ and $\mathcal {F_Y}$. 


### Modal Decomposition

Obviously, for a given PMI function, there are many ways to write it into sum of modes. We hope to have as few modes as possible. In some cases, we might even wish to compromise the precision of the sum and try to have a reasonable approximation of the given PMI with a sum of even fewer modes. That is, for a given finite $k$, we would like to solve the problem 

$$
\min_{ (\sigma_i, f_i, g_i), i=1, \ldots, k} \, \left \Vert \mathrm{PMI} - \sum_{i=1}^k \sigma_i f_i\otimes g_i\right\Vert^2
$$


This optimization is in fact a well-studied one. For the case with finite alphabets, the target PMI function can be thought as a $\vert\mathcal X\vert \times \vert\mathcal Y\vert$ matrix, with the $(x,y)$ entry being the function value $\mathrm {PMI}(x,y)$; and the above optimization problem is solved by finding the [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) of this matrix. The result is a decomposition is a diagonalization, turning the PMI matrix into the sum of orthogonal rank-1 matrices, each corresponds to one mode in our definition. Here, we will define the modal decomposition with a sequential construction, which is indeed a standard way to define SVD.   

---
**Definition: Rank-1 Approximation**

For a function $B \in \mathcal {F_{X\times Y}}$, and a given reference distribution $R_{\mathsf {xy}} = R_\mathsf x R_\mathsf y$, the rank-1 approximation of $B$ is written as an operator $\Gamma: B \mapsto (\sigma, f^\ast, g^\ast)$,  

$$
\Gamma(B) \stackrel{\Delta}{=} \arg\min_{\sigma, f, g} \; \Vert B - \sigma\cdot f\otimes g\Vert^2
$$

where the optimization has the constraints: $\sigma \geq 0$, $f^\ast \in \mathcal {F_X}, g^\ast\in \mathcal {F_Y}$, are standard feature functions, i.e., $f^\ast, g^\ast$ both have zero mean and unit variance w.r.t. $R_\mathsf{x}, R_\mathsf{y}$, respectively.

---
We will state here without proof an intuitive property of this approximation, which we will use rather frequently: the approximation error is orthogonal to the optimal feature functions, i.e. 

$$ 
\begin{align*}
&\sum_{x\in \mathcal X} \; R_{\mathsf x}(x) \cdot \left[ \left(B(x,y) - \sigma\cdot f^\ast (x) g^\ast (y)\right) \cdot f^\ast (x) \right] = 0 , \qquad \forall y\\ 
&\sum_{y\in \mathcal Y} \; R_{\mathsf y}(y) \cdot \left[ \left(B(x,y) - \sigma\cdot f^\ast (x) g^\ast (y)\right) \cdot g^\ast (y) \right] = 0 , \qquad \forall x
\end{align*}
$$

Based on this we have the following definition of modal decomposition. 

---
**Definition: Modal Decomposition $\zeta$**

For a given joint distribution $P_{\mathsf {xy}}$ on $\mathcal {X \times Y}$ and a reference distribution $R_{\mathsf {xy}} = R_\mathsf x R_\mathsf y$. We denote the rank-1 approximation of the PMI as 

$$
\zeta_1(P_{\mathsf {xy}}) = (\sigma_1, f_1^\ast, g_1^\ast) \stackrel{\Delta}{=} \Gamma (\mathrm {PMI}) = \Gamma \left(\log \frac{P_{\mathsf {xy}}}{P_\mathsf xP_\mathsf y}\right)
$$

and for $i=2, 3, \ldots$, $\zeta_i$ as the the rank-1 approximation of the approximation error of all the previous steps:

$$
\zeta_i(P_{\mathsf{xy}}) = (\sigma_i, f_i^\ast, g_i^\ast ) \stackrel{\Delta}{=} \Gamma \left(\mathrm{PMI} - \sum_{j=1}^{i-1} \sigma_j \cdot f_j^\ast \otimes g_j^\ast \right)
$$

Collectively, $\lbrace \zeta_i \rbrace : P_{\mathsf {xy}} \mapsto [(\sigma_i, f^\ast_i, g^\ast_i), i=1, 2, \ldots]$ is called the **modal decomposition operation**

---

A few remarks are in order. 

1. The following facts are similar to those of SVD, following similar proof, which we omit:
- $\sigma_1 \geq \sigma_2 \geq \ldots$ in descending order
- $\langle f^\ast_i, f^\ast_j \rangle = \langle g^\ast_i, g^\ast_j \rangle = \delta_{ij}$, i.e. the feature functions in different modes are orthogonal to each other. 

2. We denote this decomposition as $\lbrace\zeta_i \rbrace (P_{\mathsf {xy}})$, or simply $\zeta(P_{\mathsf {xy}})$, which should be read as "the $\zeta$-operation for the $\mathsf{x-y}$ dependence defined by the joint distribution $P_{\mathsf{xy}}$". While we write the functional decomposition as an L2 approximation to the PMI function, the PMI is not the unique way to describe the dependence. Later we will have examples where it is convenient to use a slightly different target function, with the resulting choices of the feature functions also a bit different. We consider all such operations to decompose the dependence as the same general idea. 
3. The definition says that for each model $P_{\mathsf {xy}}$ there is an ideal sequence of modes for the orthogonal decomposition. In practice, we do not observe either the model or the mode. We will show later that learning algorithms often try to learn an approximate version of the modes. For example, it is common to only learn the first $k$ modes, or to learn the decomposition of an empirical distribution from a finite dataset, or to have extra restrictions of the learned features due to the limited expressive power of a network, etc. In more complex problems, sometimes it might not even be clear which dependence we are trying to decompose. The purpose of defining the $\zeta$ operation is to help us to clarify what type of compromises are taken in finding a computable approximate solution to the idealized decomposition problem. 


## Properties of Modal Decomposition

There are many nice properties of this modal decomposition. The best way to see them is to go through our [survey paper](http://lizhongzheng.mit.edu/sites/default/files/documents/mace_final.pdf). On this page we will only state some of them as facts without any proof, and sometimes with intuitve but not-so-precise statements. The central point of this is to make the following statement

>Modal decomposition, the $\zeta$ operation of a model $P_{\mathsf {xy}}$, decomposes the dependence between two random variables $\mathsf x$ and $\mathsf y$ into a sequence of pairwise correlation between  features $f^\ast_i(\mathsf x)$ and $g^\ast_i(\mathsf y)$, with correlation coefficient $\sigma_i$, for $i=1, 2, \ldots$. 

This statement is important since in both learning of the model $P_{\mathsf {xy}}$ and using it for inference tasks, we no longer have to carry the entire model which is often far to complex. Instead, we can learn and use only a subset of modes. Because we have $\sigma_i$'s to quantify the strengths of these modes, we would know exactly how to choose the more important modes and how good is the resulting approximate model. 

One technical issue is the **_local assumption_**. Many nice properties and connections for the modal decomposition are asymptotic statements, proved in the limiting regime where $P_{\mathsf {xy}}, P_\mathsf x \cdot P_\mathsf y$, and $R_\mathsf x\cdot R_\mathsf y$ are all "close" to each other. Such local assumptions are indeed a fundamental concept: The space of probability distributions is not a linear vector space, but a manifold. The local assumption allows us to focus on a neighborhood which can be approximated by the tangent plane of the manifold, and hence get the geometry linearized. Details of this can be found in the literature of [information geometry](https://www.amazon.com/Information-Translations-Mathematical-Monographs-Tanslations/dp/0821843028) and [correspondence analysis](https://en.wikipedia.org/wiki/Correspondence_analysis). A quick example is that the following approximation to the PMI function is often used in our development with the assumption that the precision is acceptable. 

$$
\mathrm{PMI}(x,y) = \log \left( \frac{P_{\mathsf {xy}}(x,y)}{P_\mathsf x(x) P_\mathsf y(y)} \right)\approx \widetilde{\mathrm{PMI}}(x,y) = \frac{P_{\mathsf {xy}}(x,y) - P_\mathsf x(x) P_\mathsf y(y)}{P_\mathsf x(x) P_\mathsf y(y)}
$$

This inevitably leads to some technical details in making the mathematical statements. Different statements might require different strengths of the local assumptions, and in some cases one can even circumvent such assumptions by making a slightly different statement. To avoid leading our readers into such discussions, we will simply call all of such things the "local approximation" and assume they are given for all statement regardless of what is needed. Furthermore, we will hide the rest of these statements in a toggled block. If the reader is comfortable with our main message about decomposing the dependence, then it might be better to skip the mathematical statements and move on to the later developments faster. 

<details>
<summary>Click to read the mathematical statements </summary>


**The Conditional Expectation Operator**

The first easy thing we can do with the local approximation is to apply the Taylor approximation, $\log(x) \approx x-1$ when $x\approx 1$, to the PMI function

$$
\mathrm{PMI}(x,y) = \log \left( \frac{P_{\mathsf {xy}}(x,y)}{P_\mathsf x(x) P_\mathsf y(y)} \right)\approx \widetilde{\mathrm{PMI}}(x,y) = \frac{P_{\mathsf {xy}}(x,y) - P_\mathsf x(x) P_\mathsf y(y)}{P_\mathsf x(x) P_\mathsf y(y)}
$$

We can also just take the reference $R_\mathsf x = P_\mathsf x, R_\mathsf y= P_\mathsf y$ to further simplify things. We could have defined the modal decomposition with the target function of $\widetilde{\mathrm{PMI}}$, and choose this new reference, which can also be reasonably read as "describing the dependence". We view this as purly a choice of presentation style, and the difference happens to vanish under the local approximation. 

There is an additional interesting property of this approximated version of PMI: when viewed as an operator on the functional space it is closely related to the conditional expectation operator. 

> **Property 1:** 
> Let $B : \mathcal {F_X} \to \mathcal {F_Y}$ be defined as: for $a\in \mathcal {F_X}$, $B(a) \in \mathcal {F_Y}$ with 
> 
> $$
> \begin{align*}
> \left(B(a)\right) (y) &\stackrel{\Delta}{=} \sum_{x\in \mathcal X} \widetilde{\mathrm{PMI}}(x,y)\cdot (P_\mathsf x (x) \cdot a(x))\\
> &= \sum_{x\in \mathcal X}\frac{P_{\mathsf {xy}}(x,y) - P_\mathsf x(x) P_\mathsf y(y)}{P_\mathsf x(x) P_\mathsf y(y)} \cdot  (P_\mathsf x(x) \cdot a(x))\\
> &= \mathbb E [a(\mathsf x) | \mathsf y = y ] 
> \end{align*}
> $$
>

We write sum over $x$ in the above which of course can be turned into integral when $x$ is continuous valued. The $\widetilde{\mathrm{PMI}}$ function does not directly act on the input $a(\cdot)$, but instead needs an extra $P_\mathsf x(x)$ multiplied. This can be thought as a summation weighted by the reference distribution. 

One can also define a transpose operator $B^T: \mathcal {F_Y}\to \mathcal {F_X}$, for $b \in \mathcal {F_Y}$, 

$$
\left(B^T(b)\right)(x) = \sum_{y\in \mathcal Y}\frac{P_{\mathsf {xy}}(x,y) - P_\mathsf x(x) P_\mathsf y(y)}{P_\mathsf x(x) P_\mathsf y(y)} \cdot  (P_\mathsf y(y) \cdot b(y))=  \mathbb E[b(\mathsf y)|\mathsf x=x], \forall x.
$$
<!---
There are a number of consequences when this connection is established. 

> **Property 2: Contraction**
> 
> The conditional expectation operator is known to be a contraction, i.e. $\Vert B(a) \Vert^2 \leq \Vert a \Vert^2, \qquad \forall a \in \mathcal {F_X}$.
>
>  Equivalently, if $b = B(a)$, i.e. $b(y) = \mathbb E[a(\mathsf x)|\mathsf y=y], \forall y$, then 
>  $\mathbb E_{\mathsf y\sim P_\mathsf y}[b(\mathsf y)^2] \leq \mathbb E_{\mathsf x\sim P_\mathsf x}[a(\mathsf x)^2].$ 
>

If we now look the modal decomposition $\zeta(P_\mathsf {xy}) = [(\sigma_i, f^\ast_i, g^\ast_i), i=1, 2, \ldots]$, then we have a nice orthogonal structure. 
--->

>**Property 3: Mode Correlation**
>
> $$
> (B(f^\ast_j))(y) = \sum_x \left(\sum_i \sigma_i \cdot f^\ast_i(x) g^\ast_i(y)\right) \cdot \left(P_{\mathsf x}(x) \cdot f^\ast_j(x)\right) = \sigma_j g^\ast_j(y), \quad \forall y
> $$
>
>since $\mathbb E[f^\ast_i(\mathsf x) f^\ast_j(\mathsf x)] = \delta_{ij}$. With the same math, we also have $B^T(g^\ast_j) = \sigma_j \cdot f^\ast_j$. 
>
>That is, each $g^\ast_j$ is the image of the $B(\cdot)$ operator acting on $f^\ast_j$, scaled by the corresponding $\sigma_i$, and vice versa. 
>
>Now we have 
>
>$$
>\mathbb E_{\mathsf{x,y} \sim P_{\mathsf{x,y}}} [ f^\ast_i (\mathsf x) \cdot g^\ast_j(\mathsf y)] = \mathbb E_{\mathsf x\sim P_\mathsf x} [f^\ast_i (\mathsf x) \cdot \mathbb E[ g^\ast_j(\mathsf y)|\mathsf x]] = \mathbb E_{\mathsf x\sim P_\mathsf x} [f^\ast_i (\mathsf x) \cdot \sigma_j \cdot f^\ast_j(\mathsf x)] = \sigma_i\cdot \delta_{ij}
>$$

This result says that each feature $f^\ast_i(\mathsf x)$ is only correlated with the corresponding $g^\ast_i$ feature of $\mathsf y$, and uncorrelated with all other features. $\sigma_i$ is the correlation coefficient. Thus, the dependence between $\mathsf x$ and $\mathsf y$ is in fact written as a sequence of correlation between feature pairs, each with a strength quantified by the corresponding $\sigma_i$. 

In [HGR], the HGR maximal correlation is defined for a given joint distribution $P_{\mathsf {xy}}$ as

$$
\rho_{\mathrm{HGR}} \stackrel{\Delta}{=} \max_{f \in \mathcal {F_X}, g \in \mathcal {F_Y}} \; \rho (f(\mathsf x), g(\mathsf y)),
$$

where $\rho$ denotes the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
This maximal correlation coefficient $\rho_{\mathrm{HGR}}$ is used as a measure of dependence between the two random variables. At this point, it should be clear that the modal decomposition structure is a natural generalization. $\sigma_1$, the correlation between the strongest correlated feature pairs is exactly the HGR maximal correlation coefficient. Beyond that, there are indeed a sequence of correlated feature pairs in descending order of strengths.  

### Divergence and Fisher Information

As we stated with the [definition of modes](#a-single-mode), writing the PMI function as sum of product functions puts the model $P_{\mathsf {xy}}$ on an exponential family. Locally, the behavior of such exponential family is fully determined by the [Fisher information](https://en.wikipedia.org/wiki/Fisher_information). For example, if 
$$\mathrm{PMI}= \log\frac{P_{\mathsf{xy}}}{P_\mathsf x P_\mathsf y} = \sum_{i=1}^k f_i \otimes g_i,$$

then the conditional distribution $P_{\mathsf {x \vert y}}(\cdot \vert y)$, for different values of $y$, are on a $k$ - dimensional exponential family with $f_i(\cdot), i=1, \ldots, k$ as the natural statistics, and $g_i(y), i=1, \ldots k$ as the corresponding parameters. The Fisher information for this family is a $k\times k$ matrix $\mathcal I$, with entries

$$
[\mathcal I]_{ij} = \mathbb E_{\mathbb x \sim P_\mathsf x} \left[ \left(\frac{\partial}{\partial g_i} \mathrm {PMI}\right) \cdot \left(\frac{\partial}{\partial g_j} \mathrm {PMI}\right)\right] = \mathbb E_{\mathbb x \sim P_\mathsf x} [f_i(\mathsf x) f_j(\mathsf x)] = \langle f_i, f_j \rangle
$$

which is exactly the definition of the inner product we started with. In this context, we can also understand the [orthogonal modal decomposition](#modal-decomposition) as a special and nice case where the Fisher information matrix is diagonalized. 

There are some direct consequences of this connection. 

>**Property 4: K-L divergence**
>
> If two distribution on $\mathcal X$, $P_\mathsf x$ and $Q_\mathsf x$, are both in the neighborhood of the reference distribution $R_\mathsf x$, with $\log P_\mathsf x/Q_\mathsf x = f$, then $D(P_\mathsf x \Vert Q_\mathsf x) \approx \frac{1}{2} \Vert f\Vert^2$

where $D(P \Vert Q)$ is the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). The relation between the K-L divergence and the Fisher information can be found [here](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Fisher_information_metric). 

Applying this fact to the PMI function, we have the following statement. 

>**Property 5: Mutual Information**
>
> $\frac{1}{2} \Vert \mathrm{PMI} \Vert^2 \approx D(P_{\mathsf {xy}} \Vert P_\mathsf x P_\mathsf y) = I(\mathsf x; \mathsf y)$

where $I(\mathsf x; \mathsf y)$ is the [mutual information](https://en.wikipedia.org/wiki/Mutual_information) between $\mathsf x$ and $\mathsf y$, which is another popular way to measure how much the two random variables depend on each other. 

Now if we have the modal decomposition $\zeta(P_\mathsf {xy}) = [(\sigma_i, f^\ast_i, g^\ast_i), i=1, 2, \ldots]$, we have the following result. 

>**Property 6: Decomposition of the Mutual Information**
>
> $I(\mathsf x; \mathsf y) = \frac{1}{2} \Vert \mathrm{PMI} \Vert^2 = \frac{1}{2} \sum_i \sigma_i^2$

This is probably the cleanest way to understand the modal decomposition: it breaks the mutual information into the sum of a number of modes, as the (squared) strengths of these modes add up to the mutual information. As stated earlier, it is often difficult to learn or to store the PMI function in practice due to the high dimensionality of the data. In these cases, it is a good idea to approximate the PMI function with a truncated versition that only keeps the first $k$ strongest modes. This not only gives the best rank-limited approximation of the joint distribution, as stated in equation (2) in the [definition](#definition-modal-decomposition-zeta), but also captures the most significant dependence relation (the most strongly correlated feature pairs), and in that sense makes the approximation useful in inference tasks. 


</details>

## An Example of Numerical Computation of Modal Decomposition

To wrap up this introduction page, we will show one simple example, where we have a small synthesized dataset to train a small neural network. When the training procedure converges, we demonstrate that the learned features match with the result of the $\zeta$ operation. The purpose of this numerical example is to show that in some extreme cases, with a simple dataset and a carefully chosen neural network, the learning in the neural network is indeed finding a low rank approximation to the true model, which is consistent with the modal decomposition operation defined in this page. 

>**Note:**
>It should be noted that by this example we are simply pointing out the connection between learning in neural networks and the modal decomposition. We use that as an evidence that modal decomposition is a relevant concept in practical problems: good for inference tasks with a reasonable cost to learn from data. It is _not_ our intention to suggest that neural networks are the "recommended" numerical procedure to compute modal decomposition. In fact, the purpose of this series of pages is to show that under different sceniaros, guided by the geometric understanding of modal decomposition, we can intellegently choose different numerical methods, make adjustment or even develop new algorithms, for this purpose.  


The dataset we use is a $(x_i, y_i), i=1, \ldots, n$ drawn i.i.d. from a joint distribution $P_{\mathsf {xy}}$, where $\vert \mathcal X \vert = 8, \vert \mathcal Y \vert = 6$, and we simply randomly choose a joint distribution in this space. 

{% highlight python %}

def Generate2DSamples(xCard, yCard, nSamples):
    
    # randomly pick joint distribution, normalize
    Pxy = np.random.random([xCard, yCard])
    Pxy = Pxy / sum(sum(Pxy))

    # compute marginals
    Px = np.sum(Pxy, axis=1)
    Py = np.sum(Pxy, axis=0)    
    
    lp = np.reshape(Pxy, xCard*yCard)
        
    data = np.random.choice(range(xCard*yCard), nSamples, p=lp)
    
    X = (data/yCard).astype(np.int) 
    Y = data % yCard
    
    return([Pxy, Px, Py, X, Y])


xCard = 8
yCard = 6
nSamples = 100000

[Pxy, Px, Py, X, Y] = Generate2DSamples(xCard, yCard, nSamples)

{% endhighlight %}

Because we know the model perfectly here, we can directly compute the $\zeta$ operation with an SVD. We pick only the first pair of feature functions. Note in order to avoid the weights by the reference, we appropriately put some square roots in these code to make the SVD computation weight-free. 

{% highlight python %}

def WhatTheorySays(Pxy):
    '''
    Compute theoretical answers for f and g, corresponding to the 1st pair
    of singular vectors
    Return in normalized function form
    '''
    # compute marginals
    [xCard, yCard] = Pxy.shape
    Px = np.sum(Pxy, axis=1)
    Py = np.sum(Pxy, axis=0) 
    
    T = np.tile(np.sqrt(Px).reshape(xCard,1), [1, yCard]) \
        * np.tile(np.sqrt(Py), [xCard, 1])
    B = Pxy / T
    U, d, R = np.linalg.svd(B)
    
    S = U[:,1] / np.sqrt(Px)
    v = R[1,:] / np.sqrt(Py)
    
    return([S, v])
    
[S_theory, v_theory] = WhatTheorySays(Pxy)

{% endhighlight %}

Now we train a neural network with a single layer of 1 hidden node. That is, we allow the neural network to choose one feature function of the input $\mathsf x$, and train with cross entropy loss. After training, we take out the appropriate weights of the trained neural network, and call the "regulate()" function to make sure they are centralized and normalized. 

{% highlight python %}

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=xCard))
model.add(Dense(yCard, activation='softmax', input_dim=1))

sgd = SGD(4, decay=1e-2, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(XLabels, YLabels, verbose=0, batch_size=nSamples, epochs=100) 

weights = model.get_weights()       # pick out weights after training

S = weights[0].reshape(1, xCard)
S = expit(S + weights[1])           # this should be the f(x)'s
S = regulate(S, Px)
S = S*np.sign(sum(S*S_theory))      # make sure there is no (-1) factor
v = weights[2]                      # output weights are the g(y)'s
v = regulate(v, Py) 
v = v*np.sign(sum(v*v_theory))

{% endhighlight %}

Finally we plot the learning result from the neural network $[S, v]$ against the theoretic results. Unsurprisingly, they match very well. 

## Going Forward

The idea of modal decomposition is interesting in that it offers a clear view to the low-rank solutions to the learning and inference problems with high-dimensional data, which is clearly a central issue. It offers a clean geometric view of the problem as finding the "right" orthogonal basis for the functional space. 

From a theoretical point-of-view, the geometry helps to quantify the usefulness of a selected feature in a certain inference task. In other words, it gives a measure of "relevance" so that we can distinguish and select among information snipts, or features, based on the contents they carry. Such a way of discriminate information pieces based on the semantics is largely not addressed in the conventional information theory, but needed in learning problems. 

Operationally, the geometric operations defined in the functional space lead to a number of interesting design ideas for learning algorithms, to skip parts of the learning procedure and arrive at certain desired learning results more directly, to handle more complex learning applications involving  multiple models, constraints, or distributed learning agents. In the next a few pages in this series we will present some more aggressive and flexible designs of learning algorithms to solve some more complex problems. 

