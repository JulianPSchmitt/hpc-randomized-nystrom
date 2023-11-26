# Randomized Nyström Low-Rank Approximation
We aim to study the randomized Nyström algorithm for computing a low-rank
approximation of a large and dense matrix $A \in \mathbb{R}^{n \times n}$ that
is symmetric positive semidefinite. Given a sketching matrix $\Omega \in
\mathbb{R}^{n \times l}$, where $l$ is the sketch dimension, the randomized
Nyström approximation relies on the following formula:

$$A_{Nyst} = (A \Omega) (\Omega^T A \Omega)^+ (\Omega^T A)$$
