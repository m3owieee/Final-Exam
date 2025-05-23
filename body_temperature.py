# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:36:59 2025

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic body temperature data
np.random.seed(42)
true_mu = 36.8
true_sigma = 0.5
data = np.random.normal(true_mu, true_sigma, size=50)  

# Define prior hyperparameters
prior_mu_mean = 37  
prior_mu_precision = 1 / 0.5**2
prior_sigma_alpha = 2 
prior_sigma_beta = 0.5 

# Update the posterior parameters
posterior_mu_precision = prior_mu_precision + len(data) / (true_sigma ** 2)
posterior_mu_mean = (prior_mu_mean * prior_mu_precision + np.sum(data) / (true_sigma ** 2)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data)) ** 2) / 2

# Generate samples from the posterior distributions
posterior_mu = np.random.normal(posterior_mu_mean, np.sqrt(1 / posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

# Plot the posterior distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='pink', edgecolor='skyblue')
plt.title(r'Posterior distribution of $\mu$' '\n' 'Body Temperature in C°')
plt.xlabel(r'$\mu$ (Mean Body Temperature in C°)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='purple', edgecolor='yellow')
plt.title(r'Posterior distribution $\sigma$' '\n' 'Body Temperature in C°')
plt.xlabel(r'$\sigma$ (Standard Deviation)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Print summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print(f"Mean of mu (Body Temperature): {mean_mu:.2f}°C")
print(f"Standard deviation of mu: {std_mu:.2f}°C")

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print(f"Mean of sigma: {mean_sigma:.2f}°C")
print(f"Standard deviation of sigma: {std_sigma:.2f}°C")