"""lsbi: Linear Simulation Based Inference.

A Python library for Bayesian linear models and inference. Provides efficient
vectorized implementations of Gaussian linear models, mixture models, and 
associated statistical distributions with support for broadcasting operations.

Main Components
---------------
LinearModel : Multilinear Gaussian models for Bayesian inference
MixtureModel : Linear mixture models with categorical weights  
multivariate_normal : Vectorized multivariate normal distributions
mixture_normal : Mixture of multivariate normal distributions

Examples
--------
>>> from lsbi import LinearModel
>>> model = LinearModel(M=1, C=0.1, mu=0, Sigma=1)
>>> posterior = model.posterior([1.0, 2.0])
"""

from lsbi._version import __version__  # noqa: F401
from lsbi.model import LinearModel, MixtureModel
from lsbi.stats import multivariate_normal, mixture_normal

__all__ = [
    "__version__",
    "LinearModel", 
    "MixtureModel",
    "multivariate_normal",
    "mixture_normal"
]
