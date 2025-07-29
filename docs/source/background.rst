====================
Mathematical Background
====================

This page provides the mathematical foundation and terminology used throughout ``lsbi``. Understanding these concepts will help you use the library effectively and interpret its results correctly.

Core Linear Model
=================

The ``lsbi`` library is built around the linear Bayesian model:

.. math::

   P(D|\theta) &= \mathcal{N}(D | m + M\theta, C) \quad \text{(Likelihood)}

   P(\theta) &= \mathcal{N}(\theta | \mu, \Sigma) \quad \text{(Prior)}

Where the **posterior** distribution is computed analytically as:

.. math::

   P(\theta|D) = \mathcal{N}(\theta | \mu_{post}, \Sigma_{post})

With:

.. math::

   \Sigma_{post} &= (\Sigma^{-1} + M^T C^{-1} M)^{-1}

   \mu_{post} &= \mu + \Sigma_{post} M^T C^{-1} (D - m - M\mu)

Mathematical Notation and Terminology
=====================================

The following table defines all the mathematical symbols and their corresponding meanings in ``lsbi``:

.. list-table:: Mathematical Symbols
   :header-rows: 1
   :widths: 10 15 25 50

   * - Symbol
     - Variable Name
     - Shape
     - Description
   * - :math:`\theta`
     - ``theta``
     - ``(..., n)``
     - **Parameters** - The values you want to infer (e.g., slope and intercept of a line)
   * - :math:`D`
     - ``D``
     - ``(..., d)``
     - **Data** - Your measurements or observations
   * - :math:`M`
     - ``M``
     - ``(..., d, n)``
     - **Model Matrix** - Linear transformation from parameters to predicted data
   * - :math:`m`
     - ``m``
     - ``(..., d)``
     - **Data Offset** - Optional additive offset in data space (often zero)
   * - :math:`C`
     - ``C``
     - ``(..., d, d)``
     - **Data Covariance** - Noise and correlations in your data measurements
   * - :math:`\mu`
     - ``mu`` or ``μ``
     - ``(..., n)``
     - **Prior Mean** - Your initial best guess for the parameters
   * - :math:`\Sigma`
     - ``Sigma`` or ``Σ``
     - ``(..., n, n)``
     - **Prior Covariance** - Your uncertainty in the initial parameter guess

Key Derived Quantities
======================

From the basic model, ``lsbi`` can compute several important statistical quantities:

**Posterior Distribution**
   :math:`P(\theta|D)` - Updated belief about parameters after seeing data

**Evidence (Marginal Likelihood)**
   :math:`P(D) = \mathcal{N}(D | m + M\mu, C + M\Sigma M^T)` - Probability of observing the data

**Posterior Predictive Distribution**
   :math:`P(D_{new}|D_{old})` - Probability of new data given previous observations

**Kullback-Leibler Divergence**
   :math:`D_{KL}(P(\theta|D) || P(\theta))` - Information gain from data to posterior

Broadcasting and Shape Conventions
==================================

``lsbi`` uses NumPy broadcasting extensively. The ellipses ``...`` in the shape specifications above represent arbitrary broadcastable dimensions that allow you to:

- Analyze multiple datasets simultaneously
- Compare different model configurations
- Perform batch inference operations

**Example Broadcasting Scenarios:**

1. **Single Model, Single Dataset**: All arrays have their minimal shapes
2. **Single Model, Multiple Datasets**: ``D`` has shape ``(N, d)`` for ``N`` datasets
3. **Multiple Models, Single Dataset**: Model parameters have shape ``(K, ...)`` for ``K`` models
4. **Multiple Models, Multiple Datasets**: Both data and models are batched

Diagonal Optimizations
=====================

When covariance matrices are diagonal (uncorrelated), ``lsbi`` can use faster algorithms:

- **Diagonal Prior**: ``diagonal_Sigma=True`` when parameters are independent
- **Diagonal Data Covariance**: ``diagonal_C=True`` when data points have independent noise
- **Diagonal Model Matrix**: ``diagonal_M=True`` when each parameter affects only one data dimension

For diagonal matrices, you can pass 1D arrays containing only the diagonal elements instead of full 2D matrices, significantly improving performance for high-dimensional problems.

Common Use Cases
===============

**Linear Regression**
   Fitting lines, polynomials, or any linear combination of basis functions to data.

**Parameter Estimation**
   Inferring physical constants, calibration parameters, or model coefficients from experimental data.

**Model Comparison**
   Using the evidence :math:`P(D)` to compare different model hypotheses.

**Uncertainty Quantification**
   Computing confidence intervals and parameter correlations from the posterior covariance.

**Active Learning**
   Using the posterior predictive distribution to design optimal future experiments.