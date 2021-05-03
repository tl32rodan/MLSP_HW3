#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('config', 'IPCompleter.use_jedi=False')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# ## PCA

# In[4]:


input_array = []
for img_name in sorted(os.listdir('./four_dataset/')):
    img = Image.open(os.path.join('./four_dataset/', img_name))
    img = np.array(img).reshape(-1)
    input_array.append(img)


# In[5]:


input_array = np.array(input_array).T
input_array.shape


# In[6]:


def PCA(input_array, dim=2):
    # Normalize input array
    mu = np.mean(input_array, axis=1)
    zero_mean_input_array = input_array - np.array([mu for i in np.arange(input_array.shape[1])]).T
    
    # Use SVD to fine priciple components
    U, Sigma, V = np.linalg.svd(zero_mean_input_array)
    
    # Access top_dim priciple components
    reduced = U.T[:dim]@zero_mean_input_array
    
    # Reconstruction
    reconstructed = U[:,:dim]@reduced + np.array([mu for i in np.arange(input_array.shape[1])]).T
    
    return reconstructed


# In[12]:


fig, axs = plt.subplots(1, 5, figsize=(35, 7))
axs[0].imshow(input_array[:,0].reshape(28,28))
axs[0].set_title('Original image', fontsize=24)

for i, dim in enumerate([2,16,64,256]):
    reconstructed = PCA(input_array, dim=dim)
    axs[1+i].imshow(reconstructed[:,0].reshape(28,28))
    axs[1+i].set_title('Reconstructed image(dim='+str(dim)+')', fontsize=24)
plt.show()


# ## NMF

# In[13]:


from sklearn.preprocessing import normalize


# In[14]:


input_array = []
for img_name in sorted(os.listdir('./mixture_dataset(0147)/')):
    img = Image.open(os.path.join('./mixture_dataset(0147)/', img_name))
    img = np.array(img).reshape(-1)
    input_array.append(img)


# In[15]:


input_array = np.array(input_array).T
input_array.shape


# In[16]:


def NMF(input_array, dim=4, num_iter = 3000, delta = 1e-5):
    V = input_array
    W = np.random.rand(V.shape[0], 4)
    H = np.random.rand(4, V.shape[1])

    for idx in range(num_iter):
        # Update H
        deno = (W.T@W@H)
        deno[deno==0] = delta
        numer = (W.T@V)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                H[i,j] = H[i,j]*numer[i,j]/deno[i,j]
        # Update W
        deno = (W@H@H.T)
        deno[deno==0] = delta
        numer = (V@H.T)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i,j] = W[i,j]*numer[i,j]/deno[i,j]
                
        # To avoid NaN
        H = np.nan_to_num(H)
        W = np.nan_to_num(W)
        # Print Forbinous norm
        norm = np.linalg.norm(V - W@H, 'fro')
        if idx%100 == 0:
            print("Iter ", idx, ":", norm)
    return W, H


# In[17]:


W, H = NMF(input_array, dim=4)


# In[21]:


fig, axs = plt.subplots(1, 4, figsize=(35, 7))

for i in range(W.shape[1]):
    axs[i].imshow(W[:,i].reshape(28,28))
    axs[i].set_title('Factorized Image', fontsize=24)
plt.show()


# In[ ]:




