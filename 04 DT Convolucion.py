#!/usr/bin/env python
# coding: utf-8

# ## Discrete-time Convolution

# Convolution sum of $x[n]$ and $h[n]$:

# $x[n]*h[n]=\sum_{m=-\infty}^{\infty}x[m] \ h[n-m]$

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# In[2]:


L = 15  # length of signal x[n]
N = 10  # length of signal h[n]

x = np.ones(L)  # the rest is assumed to be zero!
plt.figure()
plt.stem(x)
h = signal.triang(N)  # the rest is assumed to be zero!
plt.figure()
plt.stem(h)
y = np.convolve(x,h)  # mode is "full" by default
plt.figure()
plt.stem(y)
plt.show()


# In[3]:


# this cell contains a code for nice ploting of the above obtained convolution (not need for the course):

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 3))
st = fig.suptitle("Convolution sum of $x[n]$ and $h[n]$", fontsize=15)
fig.subplots_adjust(wspace = 0.5, top= 0.7)

ax1.set_ylim(0,1.25)
ax1.set_xlim(-10,15)
ax1.stem(x, 'b', markerfmt='bo')
ax1.set_title('Input signal $x[n]$', fontsize=12)
ax1.text(20,0.5,r'$*$',fontsize=20)
ax1.axhline(y=0, color = 'k', linestyle='--')
ax1.axvline(x=0, color = 'k', linestyle='--')
ax1.axis('off')

ax2.set_ylim(0,1.25)
ax2.set_xlim(-10,15)
ax2.stem(h,'r', markerfmt='ro')
ax2.set_title('Impulse response $h[n]$', fontsize=12)
ax2.text(20,0.5,r'$=$', fontsize=20)
ax2.axhline(y=0, color = 'k', linestyle='--')
ax2.axvline(x=0, color = 'k', linestyle='--')
ax2.axis('off')

ax3.set_ylim(0,5.5)
ax3.set_xlim(-10,30)
ax3.stem(y,'g', markerfmt='go')
ax3.set_title('Output y[n]', fontsize=12)
ax3.axhline(y=0, color = 'k', linestyle='--')
ax3.axvline(x=0, color = 'k', linestyle='--')
ax3.axis('off')
plt.show()


# ### Convolution sum step by step

# - Steps:
#     - invert
#     - shift
#     - multiply
#     - sum

# In[4]:


# in the previous example we assumed that both signlas start exactly at zero!
# lets change this a bit:

L = 3  # length of signal x[n]
N_min=-1
n = np.arange(N_min,N_min+L)  # length of signal h[n]

x = np.ones(L)
h = np.exp(-n*0.5)
plt.figure()
plt.stem(x)
plt.figure()
plt.stem(n,h)
plt.show()


# In[5]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(n, h,'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# #### Invert $h[n]$, multiply signals and add all the obtained samples to get $y[0]$:

# In[6]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
n = n[::-1] # invert
plt.stem(n, h, 'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# #### Shift $h[n]$

# $y[-1]$:

# In[7]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)

n = n-1
plt.stem(n, h, 'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# $y[1]$:

# In[8]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)

n = n+2
plt.stem(n, h, 'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# $y[2]$:

# In[9]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)

n = n+1
plt.stem(n, h, 'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# $y[3]$:

# In[10]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)

n = n+1
plt.stem(n, h, 'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# $y[4]$:

# In[11]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)

n = n+1
plt.stem(n, h, 'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# $y[5]$:

# In[12]:


plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(x,'b', markerfmt='bo')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)

n = n+1
plt.stem(n, h, 'r', markerfmt='ro')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# In[13]:


y = np.convolve(x,h)
# time span of y[n]: np.arange(N_min,N_min+L) 
n_y=np.arange(N_min,N_min+2*L-1)

plt.figure(figsize=(5,1.5))
plt.xlim(-5,5)
plt.stem(n_y,y,'g', markerfmt='go')
plt.axhline(y=0, color = 'k', linestyle='--')
plt.axvline(x=0, color = 'k', linestyle='--')
plt.show()


# ## Lets do this in one for loop:

# In[14]:


L_h = 3  # length of signal h[n]
N_min_h=-1
n_h = np.arange(N_min,N_min+L_h)  # time for signal h[n]
h = np.exp(-n_h*0.5)


N_min_x=0
L_x=3
x = np.ones(L_x)
n_x= np.arange(N_min_x,N_min_x+L_x)  # time for signal x[n]


plt.stem(n_x,x)
plt.figure()
plt.stem(n_h,h)
plt.show()


# In[15]:


y=np.zeros(L_x+L_h-1)  # initialization of the output

for n in range(0,L_x+L_h-1):
    y[n]=0
    for k in range(0,L_x):
        if ((n-k>=0) and (n-k<L_h)):
             y[n]=y[n]+x[k]*h[n-k]

# at the end we just need to define the time axes for the output:
n_y=np.arange(N_min_x+N_min_h,N_min_x+N_min_h+L_x+L_h-1)
plt.stem(n_y,y)
plt.show()


# # Filtering example:

# In[16]:


n=np.arange(0,45,1)
x=np.cos(np.pi*0.2*n)


# In[17]:


# Add some disturbance at n=30 and n=35
x[30]=x[30]+1
x[35]=x[35]-1
plt.stem(n,x)
plt.show()


# In[18]:


# Convolution with a Moving Average filter
h=np.array([0.25, 0.25, 0.25, 0.25])
yc=np.convolve(x,h)
plt.stem(yc)
plt.show()


# In[19]:


## Exercise 1:  a) Change the Moving Average (MA) filter length to a) N=8, b) N=12
##                      b) Change frequency of the cosine


# # Filtering upsampled signal

# In[20]:


n=np.arange(0,10,1)
x=np.cos(n)
plt.stem(n,x)
plt.show()


# In[21]:


#Upsampling by factor 4  (adding 3 zeros between each two samples)
n_up=np.arange(0,40,1)
x_up=np.zeros(40)
for i in range(0,10):
    x_up[i*4]=x[i] 
plt.stem(n_up,x_up)
plt.show()


# In[22]:


y_up=np.convolve(x_up,h)
plt.stem(y_up)
plt.show()


# In[23]:


#Exrecise 2:  Filter twice the upsampled signal above using the same MA filter
#Exercise 3:  Upsample the x[n]=cos[n] by a) N=6 and filter twice using MA of lenght 6
#Exrecise 4:  Upsample the x[n]=cos[n] by a) N=10 and filter twice using MA of lenght 10
#Exrecise 5:  Add a higher frequency sin signal to a low frequency sin signal, 
#             and filter with MA filters of different sizes
#             e.g.   x[n]=cos(0.1*pi*n)+cos(0.85*pi*n)


# # Exercise 3: Convolve signals $x[n]=e^{0.5n}u[n]$ and $h[n]=u[n]$ . 

# # Exercise 4: Convolve signals $x[n]={0.8}^n u[n]$ and $h[n]=cos[\frac{\pi n}{10}]+10cos[\frac{4\pi n}{5}]$ . 

# In[ ]:




