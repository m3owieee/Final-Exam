# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:34:40 2025

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 1, 1000)
from scipy.stats import beta

heads = 15
flips = 20
alpha_post = 1 + heads
beta_post = 1 + flips - heads

posterior_pdf = beta.pdf(x, alpha_post, beta_post)
plt.figure(figsize=(8, 5))
plt.plot(x, posterior_pdf, label=f'Posterior: Beta({alpha_post}, {beta_post})', color='pink')
plt.fill_between(x, posterior_pdf, alpha=0.3, color='violet')
plt.title('Posterior Distribution of Coin Bias')
plt.xlabel('Probability of Heads (θ)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

mean_bias = alpha_post / (alpha_post + beta_post)
ci_bias = beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"Estimated bias (mean θ): {mean_bias:.3f}")
print(f"95% credible interval: [{ci_bias[0]:.3f}, {ci_bias[1]:.3f}]")
