---
layout: post
title:  "Bayesian Quadrature"
categories: jekyll update
img: bayesian_quadrature.png
---

Bayesian quadrature (BQ) is a numerical integration method that works best on low-dimensional and expensive integration tasks. 
BQ can use active learning to query the integrand at locations that give the most information about the integrand and is thus quite sample efficient.

Bayesian quadrature returns not only an estimator but a full posterior distribution on the integral value
which can be used in downstream decision-making and uncertainty analysis [[1, 2]](#refereces-on-quadrature).

Bayesian quadrature is especially useful when integrand evaluations are expensive and sampling schemes 
prohibitive, for example when evaluating the integrand involves running a complex computer simulation, a real-world experiment,
or a lab. But even when evaluation cost is manageable, the sheer amount of queries that might be required by classical 
algorithms is usually incentive enough to favor a smarter and less wasteful approach.

Instead, Bayesian quadrature draws information from the structure of the integrand such as regularity or smoothness, and is thus 
able to learn faster from fewer integrand evaluations. Such structure is encoded as prior in the BQ model.
In Emukit, the surrogate model for the integrand would be the *emulator* and the integrand itself that can be queried, would
be the *simulator*, e.g., a real system or complex computer simulation.

#### Bayesian Quadrature in the Loop

Like other sequential learning schemes, Bayesian quadrature iteratively selects points where the integrand will be queried 
next such that an acquisition function is maximized. The acquisition function is specific to the integration
task and encodes, for example, how much a *potential* integrand evaluation will inform the unknown integral value.
In turn, the probabilistic model---usually a Gaussian process---is updated with the newly collected evaluation and 
refined at each step by optimizing its hyperparameters. 
The actual integration that yields the distribution over the integral value is then performed by integrating the *emulator*
or an approximation thereof. Usually this surrogate integral is analytic and requires little computation
in comparison to queries of the integrand.

Thus, Bayesian quadrature is based on three things: i) replacing an intractable integral with a regression 
problem on the integrand function ii) replacing the actual integration with an easier, analytic integration of the emulator
on the integrand function, and iii) actively choosing locations for integrand evaluations such that the budget is optimally used
in the sense encoded by the acquisition scheme.

#### Bayesian Quadrature in Emukit
Emukit, among others, provides functionality for vanilla Bayesian quadrature where a Gaussian process surrogate model is placed upon 
the integrand which is then integrated directly. 
This is how it is done:

First we define the function that we want to integrate. It is called `user_function` in the code block below. 
Here we choose the 1-dimensional
Hennig1D function which is already implemented in Emukit 
(see [here](https://nbviewer.org/github/emukit/emukit/blob/main/notebooks/Emukit-tutorial-Bayesian-quadrature-introduction.ipynb) 
for a visualization). We also choose the integration bounds: A lower bound and an upper bound.

```python
from emukit.test_functions.quadrature import hennig1D

user_function = hennig1D()[0]
lb = -3. # lower integral bound
ub = 3. # upper integral bound
```

Next we choose three locations for some initial evaluations to get an initial model of the integrand, also called the initial design.
Here we use the GP regression model of [GPy](https://github.com/SheffieldML/GPy) since a wrapper already exists in Emukit. Note that in BQ we are usually restricted
in the choice of the kernel function. Emukit supports a couple of different kernels such as the RBF kernel used here.

```python
import numpy as np
import GPy

X = np.array([[-2.],[-0.5], [-0.1]])
Y = user_function.f(X) # inital integrand evaluations at locations X 
gpy_model = GPy.models.GPRegression(X=X, Y=Y, 
                                    kernel=GPy.kern.RBF(input_dim=X.shape[1], 
                                    lengthscale=0.5, 
                                    variance=1.0))
```

Now we convert the [GPy](https://github.com/SheffieldML/GPy) GP model into an Emukit quadrature GP. 
Note that we also need to wrap the RBF kernel of the GPy model since Bayesian quadrature essentially integrates the kernel function. 
We integrate with respect to the Lebesgue measure whose domain is defined by the integral bounds.

```python
from emukit.model_wrappers.gpy_quadrature_wrappers import \
    BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
from emukit.quadrature.measures import LebesgueMeasure

emukit_rbf = RBFGPy(gpy_model.kern)
emukit_measure = LebesgueMeasure.from_bounds(bounds=[(lb, ub)])
emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_rbf, emukit_measure)
emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
```

Among others, Emukit supports vanilla Bayesian quadrature where the GP model directly emulates the 
integrand function. Other approaches may emulate a transformation the of the integrand (see e.g., [[3]](#refereces-on-quadrature)).

```python
from emukit.quadrature.methods import VanillaBayesianQuadrature

emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X, Y=Y)
```

Now we define the active learning loop. The essential piece in the loop is the acquisition function. The vanilla BQ loop 
by default uses the integral-variance-reduction acquisition (IVR) which measure how much the uncertainty about the integral value 
will shrink were we to evaluate the integrand at a certain location. The next evaluation of the integrand will be chosen such 
that the IVR is maximized. IVR is a global quantity of the space meaning that it takes into account what we
learn about other locations in space if we evaluate at a local point.

```python
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop

emukit_loop = VanillaBayesianQuadratureLoop(model=emukit_method)
```

Finally, we run the loop for `num_iter = 20` iterations. This will collect 20 additional integrand evaluations, chosen by 
optimizing the acqusition function at every step. After each newly collected evaluation, the vanilla BQ model is updated 
and fitted to the new dataset.

```python                           
num_iter = 20          
emukit_loop.run_loop(user_function, stopping_condition=num_iter)
```

And that's it! You can retrieve the integral and variance estimator by running
 
```python
integral_mean, integral_variance = emukit_loop.model.integrate()
``` 



Check our list of [notebooks](https://nbviewer.org/github/emukit/emukit/blob/main/notebooks/index.ipynb) and 
[examples](https://github.com/amzn/emukit/tree/master/emukit/examples) if you want to learn more about how to do Bayesian 
quadrature and other methods with Emukit. You can also check the Emukit [documentation](https://emukit.readthedocs.io/en/latest/).

We’re always open to contributions! Please read our [contribution guidelines](https://github.com/amzn/emukit/blob/master/CONTRIBUTING.md) for more information. We are particularly interested in contributions
regarding examples and tutorials.

#### Refereces on Quadrature

- [1] O'Hagan (1991) [Bayes-Hermite Quadrature](https://www.sciencedirect.com/science/article/pii/037837589190002V), *Journal of Statistical Planning and Inference* 29, pp. 245--260.
- [2] Diaconis (1988) [Bayesian numerical analysis](http://probabilistic-numerics.org/assets/pdf/Diaconis_1988.pdf), *Statistical decision theory and related topics* V, pp. 163--175.
- [3] Gunter et al. (2014) [Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature](https://papers.nips.cc/paper/5483-sampling-for-inference-in-probabilistic-models-with-fast-bayesian-quadrature), *Advances in Neural Information Processing Systems*, 27, pp. 2789--2797.
