# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:48:34 2025

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Prior parameters (uninformative)
alpha_prior = 1
beta_prior = 1

# Observed data: 20 out of 25 students submitted on time
successes = 20
trials = 25

# Posterior parameters
alpha_post = alpha_prior + successes
beta_post = beta_prior + trials - successes

# Posterior distribution
x = np.linspace(0, 1, 1000)
posterior_pdf = beta.pdf(x, alpha_post, beta_post)

# Plot the posterior distribution
plt.figure(figsize=(8, 5))
plt.plot(x, posterior_pdf, color='pink', label=f'Posterior: Beta({alpha_post}, {beta_post})')
plt.fill_between(x, posterior_pdf, alpha=0.3, color='violet')
plt.title('Posterior Distribution of Homework Submission Rate')
plt.xlabel('Probability of On-Time Submission (θ)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Summary statistics
mean_theta = alpha_post / (alpha_post + beta_post)
ci_theta = beta.ppf([0.025, 0.975], alpha_post, beta_post)

print(f"Estimated submission rate (mean θ): {mean_theta:.3f}")
print(f"95% credible interval: [{ci_theta[0]:.3f}, {ci_theta[1]:.3f}]")
