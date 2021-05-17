#!/usr/bin/env python
# coding: utf-8

# # Generate Regression Data

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import torch


# In[4]:


def make_lin_regression(n_features = 1, n_samples = 100, noise = 10, show_plot = True):
    X, y = make_regression(
        n_samples = n_samples,
        n_features = n_features,
        noise = noise,
    )
    dom_np = np.linspace(X.min(), X.max(), 20)
    dom = torch.from_numpy(dom_np).unsqueeze(-1).float()
    if show_plot is True:
        fix, ax = plt.subplots()
        ax.plot(X, y, ".")
    return dom_np, dom, X, y


# In[10]:


#make_lin_regression()

