# ---
# title: "Neural Controlled Differential Equations in Computer Vision"
# collection: teaching
# permalink: /ML/NeuralCDEforCV
# date: 2024-11-01
# ---

**Note: This blog is still in progress. Some later parts are not yet finished.**

# Table of Contents
1. Introduction
2. What are Neural Differential Equations?
3. Neural CDEs for Mask Propagation: Theory
4. Neural CDEs fro Mask Propagation: Application
	1. The Dataset
	2. Implementation of the Neural CDE Model
	3. Some Visual Examples
	4. Discussion: Advantages and Limitations of Neural CDEs


## TLDR
Neural Differential Equations provide a continuous-time model for time series data, allowing one to interpolate between data points and extrapolate beyond the training data quite well. 
In Computer Vision they can be used for a natural interpolation of frames or for a continuous propagation of masks in video data. Code available on [Github](https://github.com/fekonrad/NeuralCDE_MaskPropagation)
 
## Introduction
Recently I read about the model class of *Neural Differential Equations*. The basic idea behind them is to parametrize the driving vector field $f$ of a differential equation by a (Deep) Neural Network $f_\theta$, then solve the corresponding differential equation numerically to obtain a prediction. Since the corresponding model is a continuous-time model, i.e. we can evaluate it at any arbitrary positive time point $t\in\mathbb R_{\geq 0}$, neural differential equations work quite well for sequential data which may possibly be sampled at irregularly spaced time points. 

Lately, there have been a few papers attempting to use neural differential equations in computer vision, for example both ["Vid-ODE: Continuous-Time Video Generation with Neural Ordinary Differential Equation"](https://arxiv.org/abs/2010.08188) and ["Simple Video Generation using Neural ODEs"](https://arxiv.org/abs/2109.03292) define an Encoder-Decoder architecture, training a Neural Ordinary Differential Equation in the Latent Space in order to interpolate or extrapolate frames in video data. 

An example of Neural *Controlled* Differential Equation being used in Computer Vision consider the recent work ["Exploiting Inductive Biases in Video Modeling through Neural CDEs"](https://arxiv.org/pdf/2311.04986.pdf). The goal of this blog post is to present and implement some of the contents of this paper to give an insight of why neural differential equations might be useful for computer vision tasks. 


## What are Neural Differential Equations?
In this section I will provide a more detailed explanation of what Neural Differential Equations are and how they are trained.
But first, should you want to read more about this topic, I'll refer you to [Kidger's thesis "On Neural Differential Equations"](https://arxiv.org/pdf/2202.02435.pdf), which serves as an excellent and comprehensive introduction.  

**The Idea:**
Suppose we are given multivariate time series data $\hat y_{t_0},\hat y_{t_1},\dots, \hat y_{t_k}$, where $\hat y_{t_i}\in\mathbb R^d$ denotes the observation at  time $t_i$. We do not assume anything about the time points $t_i$, which may possibly be irregularly spaced. 
Our goal is to find a continuous-time process $(y(t))_{t\geq 0}$ which (a) fits the observations $\hat y_{t_i}$ and (b) is able to reasonably interpolate and extrapolate to new time points not present in the observational data, i.e. to time points $t\in (t_i,t_{i+1})$ (interpolation) or  $t>t_k$ (extrapolation). We do so by considering a differential equation for the process $(y(t))_{t\geq 0}$ 

The differential equation in question can take on different forms: 
1. Neural *Ordinary* Differential Equations (NODEs): $y'(t)=f_\theta(t,y(t))dt$ 
2. Neural *Controlled* Differential Equations (NCDEs): $y'(t)=f_\theta(t,y(t))dx(t)$ for some control path $(x(t))_{t\geq 0}$ 
3. Neural *Stochastic* Differential Equations (NSDEs): $dy_t=\mu_\theta(t,y_t)dt+\sigma_\theta(t,y_t)dB_t$ for a Brownian Motion $B$

where in each case, $f_\theta,\mu_\theta,\sigma_\theta$ are neural networks with parameters $\theta$. 
NCDEs are very similar to NODEs, in the sense that if the control path $(x(t))_{t\geq 0}$ is differentiable, the controlled differential equation reduces to an ordinary differential equation:

$$y'(t)=\underbrace{f_\theta(y(t))dx(t)}_{\text{CDE}}=\underbrace{f_\theta(y(t))\cdot x'(t)dt}_{\text{ODE}}$$

Since in most applications, the control path $x$ is only given by finitely many data points $x(t_1),\dots, x(t_k)$, we have some freedom in choosing the continuous-time interpolation $(x(t))_{t\geq 0}$ of these data points. Thus we are able to always choose an interpolation which is continuously differentiable, reducing the CDE to an ODE, allowing us to use one of the many numerical solvers designed for ODEs to solve our CDE. The interpolation that seems to be often chosen, and the one which we will use here as well, is a *cubic spline interpolation*. 

*Solving the ODE (or CDE)* is then done numerically, meaning we obtain a discretized solution $y_0,y_1,\dots,y_k.$ There are various numerical solvers for ODEs, but to give a concrete example one may apply a simple Euler's scheme to the ODE $y'(t)=f_\theta(y(t))dt$ obtaining a sequence 

$$y_{n+1}=y_n + f_\theta(y_n)\cdot\tau, \hspace{1cm}y_0=y(0)$$

where $\tau$ is a step size chosen in advance. As one might be able to see, this particular numerical scheme essentially makes our neural ODE into a Residual Neural Network (ResNet)!
As a consequence, neural ODEs can be seen as a "continuous-time version" of ResNets. However, there are more sophisticated numerical solvers for ODEs, which may then result in an architecture that is not a ResNet. 

The resulting discretized solution $y_0,\dots, y_N$ is then compared to the observations $\hat y_{t_0}, \hat y_{t_1},\dots,\hat y_{t_k}$ via a loss function, e.g. mean squard error.  

**Backpropagation** can be somewhat tricky here. We essentially need to find a way to backpropagate through the ODE solver. One option is to simply backpropagate through the operations of the given ODE solver. This method is a so-called *discretize-then-optimize* method. 
Its advantage is its speed and accuracy, however since we need to store all of the values $y_0,\dots, y_N$ to perform this method, this approach can be very memory-expensive for high-dimensional data and large $N$. 
Some clever ways have been developed to reduce the memory cost, such as the *adjoint method* (which is then called *optimize-then-discretize*). 
For a good overview of backpropagation through ODE Solvers, take a look at [Kidger's Thesis, Chapter 5](https://arxiv.org/pdf/2202.02435.pdf).


## Neural CDEs for Mask Propagation: Theory
*Note: This section primarily builds off of the recent paper [Exploiting Inductive Biases in Video Modeling through Neural CDEs](https://arxiv.org/pdf/2311.04986.pdf).*

First, let's try to get an intuitive idea of one might think of applying neural differential equations for the task of mask propagation: **TODO** 

Now let's get to the question of how one may apply neural differential equations to sequential image data, e.g. video data: We will consider the task of propagating masks for a sequence of input images, meaning: Given a sequence of input frames $x_{t_0}, x_{t_1}, \dots, x_{t_k}\in\mathbb R^{c\times h\times w}$ and an initial mask $y_{t_0}\in\mathbb R^{k\times h\times w}$, where $k$ is the number of classes (in the case of binary segmentation, i.e. $k=2$ we will just consider $y_{t_0}\in\mathbb R^{h\times w}$), our goal is to "propagate" the mask $y_{t_0}$ to the frames $x_{t_1},\dots, x_{t_k}$ meaning to continuously change the initial mask $y_{t_0}$ in order to match the subsequent frames $x_{t_i}$. 

For this, the authors of ["Exploiting Inductive Biases in Video Modeling through Neural CDEs"](https://arxiv.org/pdf/2311.04986.pdf) propose to model the propagation of masks via a *Controlled Differential Equation* (in a latent space). The basic structure of their algorithm is as follows:
1. **Encode** the input sequence $(x_{t_i})_{i=0,1,\dots,k}$ into a latent space: $\mathcal E_\theta(x_{t_i})=v_i$ 
2. **Interpolate** the latent vectors $(v_i)_{i=0,1,\dots, k}$ to obtain a continuous path $(V(t))_{t\geq 0}$. The Interpolation can be done by using Cubic Splines. 
3. **Solve the Neural CDE** $dz(t)=f_\theta(t,z(t))dV(t)=f_\theta(t,z(t))V'(t)dt$ in the latent space, yielding a process $(z(t))_{t\geq 0}$. 
4. **Discretize and Decode** the process $z$, i.e. compute $y_{t_i}=\mathcal D_\theta(z_{t_i})$ for the time points $t_1,\dots, t_k$. 
The resulting $y_{t_1},\dots, y_{t_k}$ will then be our propagated masks! 
Hence, the model consists of three parts that need to be trained: The encoder $\mathcal E_\theta$, the vector field $f_\theta$ and the decoder $\mathcal D_\theta$. 

For the task of mask propagation, the authors propose a "U-Net-like" architecture, where the encoder $\mathcal E_\theta$, the vector field $f_\theta$ and the decoder $\mathcal D_\theta$ are fully convolutional; with the encoder additionally performing multiple downsampling steps and the decoder performing multiple upsampling steps. 
Because of this similarity, and because neural CDEs are continous-time models, this architecture may be seen as a "*continuous-time U-Net*".


## Neural CDEs fro Mask Propagation: Application
Now let's implement the model and train it on an example dataset:

### The Dataset

### Implementation of the Neural CDE Model
The rough structure of the model will then look something like this:
```Python
class CDE_UNet(nn.Module): 
	def __init__(self): 
		self.encoder = self.Encoder() 
		self.neural_cde = self.NeuralCDE() 
		self.decoder = seld.Decoder() 
	
	# ... implementation of encoder, decoder and neural CDE ... 

	def forward(self, x, times): 
	"""
	 :param x: tensor with shape (batch_size, frames, c, h, w)
	 :param times: one-dimensional tensor with strictly increasing entries
	"""
		z = self.encoder(x)                     # encode input sequence into latent space 
		spline = CubicSplineInt(z, times)       # compute interpolating spline in latent space
		cde_sol = cdesolve(self.neural_cde, spline, times)    # solve CDE in latent space
		return self.decoder(cde_sol)            # decoder solution of CDE 
```

The structure of the encoder and decoder are essentially the same as in the standard U-Net: The encoder consists of a sequence of convolutional blocks, each followed by downsampling (max-pooling); The decoder consists of a sequence of convolutional blocks, each followed by upsampling. 
```Python 
def conv_block(in_channels, out_channels, kernel_size=3):   
    return nn.Sequential(  
        nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),  
        nn.ReLU(),  
        nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),  
        nn.ReLU()  
    )

class Encoder(nn.Module): 
	def __init__(self, in_channels=1):
		super().__init__()
		self.down1 = self.first_block(in_channels,  64)  
		self.down2 = self.conv_block(64, 128)  
		self.down3 = self.conv_block(128, 256)  
		self.down4 = self.conv_block(256, 512)  
	def forward(self, x): 
		
class Decoder(nn.Module): 
	# ...

```

The vector field of the CDE is fully convolutional and essentially corresponds to the convolutional block at the "bottom" of the U-Net when passing from the encoder to the decoder. 
In general, the neural network describing the vector field tends to be quite simple (meaning not having a large amount of parameters) when working with neural differential equations. The reason for that is that when numerically solving the differential equation, we need to evaluate the vector field (neural network) multiple times, leading to quite expensive calculations if the neural network is large.

```Python
import torch.nn as nn
from torchdiffeq import odesolve 


class NeuralCDE(nn.Module) 
	def __init__(self): 
		# ... 
		
	def forward(self, x): 
		# ... 

def cdesolve(func, spline, x0, times): 
	# TODO: 
	return 0

```
### Some Visual Examples

### Discussion: Advantages and Limitations of Neural CDEs
