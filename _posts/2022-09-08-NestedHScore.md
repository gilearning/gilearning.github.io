---
title: "The Nested H-Score Network"
layout: post
---

+ a
+ b
* TOC
{:toc}

> ### The Key Points
> In our previous posts, we developed a geometric view of the feature selection problem. We started with the geometry of functional spaces by defining inner products and distances, and then relating the task of finding information carrying features to these geometric quantities. Based on this approach, we proposed the H-Score networks as one method to learn the informative feature functions from data using neural networks. Here, we describe a new architecture, the _nested H-Score network_, which is used to make projections in the functional space with neural networks. Projections are perhaps the most fundamental geometric operations, which are now made possible, and efficient, through the training of neural networks. We will show by some examples how to use this method to regulate the feature functions, incorporate external knowledge, prioritize or separate information sources, which are the critical step towards multi-variate and distributed learning problems. 


|![test image](/assets/Hscorenetwork.png){: width="250" }|
|<b> H-Score Network </b>|

## Previously
In [this page](https://gilearning.github.io/HScore/), we defined H-Score network as shown in the figure, where the two sub-networks are used to generate features of $\mathsf x$ and $\mathsf y$: $\underline{f}(\mathsf x), \underline{g}(\mathsf y)\in \mathbb R^k$. The two features are used together to evaluate a metric, the H-score, 

$$
H(\underline{f}, \underline{g}) =\mathrm{trace} (\mathrm{cov} (\underline{f}, \underline{g})) - \frac{1}{2} \mathrm{trace}(\mathbb E[\underline{f} \cdot \underline{f}^T] \cdot \mathbb E[\underline{g} \cdot \underline{g}^T])
$$

where the covariance and the expectation are evaluated by the empirical averages over the batch of samples. We use back-propagation to train the two networks to find 

$$
\underline{f}^\ast, \underline{g}^\ast = \arg\max_{\underline{f}, \underline{g}} \; H(\underline{f}, \underline{g}). 
$$

The resulting optimal choice $\underline{f}^\ast, \underline{g}^\ast$ are promised to be a good set of feature functions. They are the solutions of the **approximated** and **unconstrained** version of the [modal decomposition](https://gilearning.github.io/ModalDecomposition/) problem, which has connection to a number of theoretical problems, and, in a short sentence, picks "informative" features.  

## The Constraints
The H-Score network is an approximation of modal decomposition because of the imperfect convergence and the limited expressive power of the neural networks, the randomness in the empirical averages due to the finite batch of samples, and the local approximation we used when formulating the [modal decomposition](https://gilearning.github.io/ModalDecomposition/).

More importantly, the H-Score network generates unconstrained features. In the definition of modal decomposition, the ideal modes $(\underline{f}^\ast, \underline{g}^\ast) = \{(f^\ast_i, g^\ast_i), i=1, \ldots k \}$ satisfy: 

* normalized: $\mathrm{var}[f_i^\ast (\mathsf x)] = \mathrm{var}[g_i^\ast(\mathsf y)] = 1, \; i=1, \ldots, k$;
* orthogonal: $\mathbb E[f_i^\ast(\mathsf x)f_j^\ast(\mathsf x)] = \mathbb E[g_i^\ast(\mathsf y)g_j^\ast(\mathsf y)] =0 , \; \forall i \neq j$
* in descending order: $\sigma_i = \mathbb E[f^\ast_i(\mathsf x)g^\ast_i(\mathsf y)]$ non-increasing as $i$ increases. 

These regulations are all desirable properties in practice. Unfortunately, the H-score network generates features that in general without these constraints. This can be easily seen from the definition: for any invertible matrix $A \in \mathbb R^{k\times k}$, 

$$
H(\underline{f}, \underline{g}) = H(A\underline{f}, A^{-1} \underline{g})
$$

That is, any linear combination of the $k$ feature functions in $\underline{f}$ can be cancelled by a corresponding inverse linear transform on $\underline{g}$, which result in an equivalent solution. Thus H-score cannot distinguish between such solutions. As a result, the best thing we can hope is to get a collection of feature functions that span the same subspace as the optimal choices for the modal decomposition problem. 

The idea of viewing the subspace spanned by a collection of features as the carrier of information is itself worth some discussions. In this page, however, we will focus on how to modify the H-Score network to enforce some of these constraints. It turns out that the key is to have a new operation to make **projections** of feature functions. 

## Nested H-Score

We start by consider the following problem to extract a single mode from a given model $P_{\mathsf {xy}}$, but with a simple constraint: for a given function $\bar{f} : \mathcal X \to \mathbb R$, we would like to find a mode, i.e. a pair of features $f(\cdot), g(\cdot)$, as the optimal rank-$1$ approximation as before, but under the constraint that $f \perp \bar{f}$, i.e. $\mathbb E_{\mathsf x \sim P_\mathsf x}[f(\mathsf x) \cdot \bar{f}(\mathsf x)] = 0$:

$$
(f^\ast, g^\ast) = \arg\min_{\small{\begin{array}{l}(f, g): f\in \mathcal {F_X}, g \in \mathcal {F_Y}, \\ \qquad \quad \mathbb E[f(\mathsf x)\cdot \bar{f}(\mathsf x)]=0\end{array}}} \; \left\Vert \mathrm{PMI} - f \otimes g \right\Vert^2
$$

|![test image](/assets/nested H2.png){: width="450" }|
|<b> Nested H-Score Network to find features orthogonal to a given $\bar{f}$ </b>|


## Going Forward

 

