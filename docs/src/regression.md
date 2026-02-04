# 1-D Regression

In this example, we use the node for a regression problem. The generative model for standard GP is

$$\begin{aligned}
p(\bm{y},\bm{f}) &= p(\bm{f}) \, \prod_{i=1}^N p(y_i\, | \,f_i) \\
p(y_i \, | \, f_i) &= \mathcal{N}(y_i|f_i, \sigma^2)\\
p(\bm{f}) &= \mathcal{N}(\bm{f}| \bm{0}, \bm{K}_{ff})
\end{aligned}$$


