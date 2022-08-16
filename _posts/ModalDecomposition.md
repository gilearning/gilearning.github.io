## Modal Decomposition of Statistical Dependence



> ### The key points
>_Statistical dependence_ is the reason that we can guess the value of one random variable based on the observation of another. The is the basis of most inference problems like decision making, estimation, prediction, classification, etc. 
>
>It is, however, a somewhat ill-posed question to ask "how much does a random variable $\mathsf x$ depend on another random variable $\mathsf y$". 
>It turns out that in general statistical dependence should be understood and quantified as a high dimensional relation: two random variables are dependent through a number of **_orthogonal modes_**; and each mode can have a different "strength". 
>
>The goal of this page is to define these modes mathematically, explain why they are important in practice, and show by examples that many statistical concepts and learning algorithms are directly related to this modal decomposition idea. With that we will also build the mathematical foundation and notations for the more advanced processing using modal decomposition in the later pages. 
