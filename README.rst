=======================================
lsbi: Linear Simulation Based Inference
=======================================
:lsbi: Linear Simulation Based Inference
:Author: Will Handley & David Yallup
:Version: 0.12.3
:Homepage: https://github.com/handley-lab/lsbi
:Documentation: http://lsbi.readthedocs.io/

.. image:: https://github.com/handley-lab/lsbi/actions/workflows/unittests.yaml/badge.svg?branch=master
   :target: https://github.com/handley-lab/lsbi/actions/workflows/unittests.yaml?query=branch%3Amaster
   :alt: Unit test status
.. image:: https://github.com/handley-lab/lsbi/actions/workflows/build.yaml/badge.svg?branch=master
   :target: https://github.com/handley-lab/lsbi/actions/workflows/build.yaml?query=branch%3Amaster
   :alt: Build status
.. image:: https://codecov.io/gh/handley-lab/lsbi/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/handley-lab/lsbi
   :alt: Test Coverage Status
.. image:: https://readthedocs.org/projects/lsbi/badge/?version=latest
   :target: https://lsbi.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://badge.fury.io/py/lsbi.svg
   :target: https://badge.fury.io/py/lsbi
   :alt: PyPi location
.. image:: https://anaconda.org/handley-lab/lsbi/badges/version.svg
   :target: https://anaconda.org/handley-lab/lsbi
   :alt: Conda location
.. image:: https://zenodo.org/badge/705730277.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.10009816
   :alt: Permanent DOI for this release
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/handley-lab/lsbi/blob/master/LICENSE
   :alt: License information


``lsbi`` is a Python package for efficient Bayesian inference with models that are linear in their parameters. It is built on NumPy and leverages vectorization to allow for powerful broadcasting and performance optimizations.

What is Linear Simulation Based Inference?
------------------------------------------

``lsbi`` solves Bayesian inference problems where the data likelihood and parameter prior are Gaussian:

- **Likelihood**: :math:`P(D|\theta) = \mathcal{N}(D | m + M\theta, C)`
- **Prior**: :math:`P(\theta) = \mathcal{N}(\theta | \mu, \Sigma)`

Here, :math:`\theta` are the model parameters and :math:`D` is the data. The library provides tools to compute the posterior :math:`P(\theta|D)`, the evidence :math:`P(D)`, and other key statistical quantities analytically.

Features
--------

- **Vectorized Operations**: Perform inference on thousands of models simultaneously using NumPy broadcasting.
- **Performance Optimizations**: Native support for diagonal covariance matrices for significant speed-ups.
- **Complete Bayesian Toolkit**: Analytically compute the posterior, evidence, and posterior predictive distributions.
- **Mixture Models**: Extend linear models to handle complex, multi-modal distributions.
- **Interoperability**: A clean API built on top of NumPy arrays.

Quick Start
-----------

Here is a 2-minute example of fitting a straight line (:math:`y = mx + c`) to data.

.. code-block:: python

    import numpy as np
    from lsbi import LinearModel
    import matplotlib.pyplot as plt

    # 1. Define the true model and generate mock data
    np.random.seed(0)
    theta_true = np.array([2.0, -1.0])  # [slope, intercept]
    x_data = np.linspace(0, 1, 10)
    y_data_true = theta_true[0] * x_data + theta_true[1]
    y_noise_std = 0.1
    y_data_noisy = y_data_true + np.random.normal(0, y_noise_std, size=x_data.shape)

    # 2. Set up the Bayesian Linear Model in lsbi
    # Our parameters are theta = [slope, intercept]
    # The model is D = M @ theta, where D is y_data and M maps theta to y_data.
    M = np.vstack([x_data, np.ones_like(x_data)]).T
    
    # Priors: Broad Gaussian priors on slope and intercept
    mu = np.zeros(2)      # Prior mean
    Sigma = np.eye(2) * 4 # Prior covariance
    
    # Likelihood: The data covariance C is the noise variance
    C = np.eye(10) * y_noise_std**2

    model = LinearModel(M=M, mu=mu, Sigma=Sigma, C=C)

    # 3. Compute the posterior given the noisy data
    posterior = model.posterior(y_data_noisy)

    print("True parameters:", theta_true)
    print("Posterior mean:", posterior.mean)
    print("Posterior covariance:")
    print(posterior.cov)
    
    # 4. Plot the results
    plt.errorbar(x_data, y_data_noisy, yerr=y_noise_std, fmt='o', label='Data')
    plt.plot(x_data, y_data_true, 'k-', label='True Line')
    
    # Plot some lines drawn from the posterior
    for i in range(100):
        theta_sample = posterior.rvs()
        plt.plot(x_data, M @ theta_sample, 'r-', alpha=0.1)
    
    plt.title("LSBI Fit to Noisy Data")
    plt.legend()
    plt.show()

Installation
------------

``lsbi`` can be installed via pip

.. code:: bash

    pip install lsbi

via conda

.. code:: bash

    conda install -c handley-lab lsbi

or via the github repository

.. code:: bash

    git clone https://github.com/handley-lab/lsbi
    cd lsbi
    python -m pip install .

You can check that things are working by running the test suite:

.. code:: bash

    python -m pytest
    black .
    isort --profile black .
    pydocstyle --convention=numpy lsbi


Dependencies
~~~~~~~~~~~~

Basic requirements:

- Python 3.6+
- `anesthetic <https://pypi.org/project/anesthetic/>`__

Documentation:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__

Tests:

- `pytest <https://pypi.org/project/pytest/>`__

Documentation
-------------

Full Documentation is hosted at `ReadTheDocs <http://lsbi.readthedocs.io/>`__.  To build your own local copy of the documentation you'll need to install `sphinx <https://pypi.org/project/Sphinx/>`__. You can then run:

.. code:: bash

    python -m pip install ".[all,docs]"
    cd docs
    make html

and view the documentation by opening ``docs/build/html/index.html`` in a browser. To regenerate the automatic RST files run:

.. code:: bash

    sphinx-apidoc -fM -t docs/templates/ -o docs/source/ lsbi/

Citation
--------

If you use ``lsbi`` to generate results for a publication, please cite
as: ::

   Handley et al, (2024) lsbi: Linear Simulation Based Inference. 

or using the BibTeX:

.. code:: bibtex

   @article{lsbi,
       year  = {2023},
       author = {Will Handley et al},
       title = {lsbi: Linear Simulation Based Inference},
       journal = {In preparation}
   }


Contributing
------------
There are many ways you can contribute via the `GitHub repository <https://github.com/handley-lab/lsbi>`__.

- You can `open an issue <https://github.com/handley-lab/lsbi/issues>`__ to report bugs or to propose new features.
- Pull requests are very welcome. Note that if you are going to propose major changes, be sure to open an issue for discussion first, to make sure that your PR will be accepted before you spend effort coding it.


Questions/Comments
------------------
