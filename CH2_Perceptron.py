# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Perceptron Algorithm

# ## 2.3.1 AND gate 구현

def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=(x1*w1)+(x2*w2)
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1


AND(1,0)

AND(1,1)

AND(0,0)

AND(0,1)

# ## 2.3.2 weight and bias

import numpy as np
x=np.array([0,1]) #input
w=np.array([0.5,0.5]) #weight
b=-0.7 #bias

w*x

np.sum(w*x)

np.sum(w*x)+b


# ## 2.3.3 weight와 bias를 이용한 gate 구현

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7 #bias
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1


AND(1,2)

# +
import numpy as np
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1
    
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1



# -

NAND(1,1)

NAND(1,0)

OR(1,1)

OR(1,0)

OR(0,0)


def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y


XOR(0,0)

XOR(1,0)

XOR(0,1)

XOR(1,1)


