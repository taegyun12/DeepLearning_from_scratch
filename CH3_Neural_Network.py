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

# # Neural_Network

# ## 3.2 activation function (활성화 함수)
# ### -입력 신호의 총합을 출력신호로 변환하는 함수를 말한다. 
# ### -perceptron에서는 활성화 함수로 step function(계단함수)를 이용하는데, 계단함수란 경계값을 기준으로 출력이 바뀌는 함수를 말한다.
#

# ## 3.2.1 Sigmoid function (시그모이드 함수)

# ### 신경망에서는 계단함수가 아닌 다른 함수들을 활성화 함수로 사용하는데, 그 중 한 예시가 시그모이드 함수이다.
# ### h(x) = 1 / 1 + exp(-x)

# ## 3.2.2 step function 구현하기
#

#입력이 0을 넘으면 1을 출력, 그 외에는 0을 출력
def step_function(x):   # x에는 실수만 입력 가능, numpt의 array등은 입력 불가
    if x > 0:
        return 1
    else:
        return 0


#계단함수의 매개변수로 numpy의 array도 받을 수 있게 함수 수정
import numpy as np
def step_function(x):
    y = x > 0
    return y.astype(np.int64)


x = np.array([-1.0,1.0,2.0])
x

y = x>0
y # bool type 의 배열이 리턴된다

y=y.astype(np.int64)
y

# ## 3.2.3 step function 그리기

# +
import matplotlib.pylab as plt

x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


# -

# ## 3.2.4 Sigmoid function 구현하기

# +
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([-1.0,1.0,2.0])
sigmoid(x)
# -

x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


# ## 3.2.7 ReLU function
#
# ### 입력이 0을 넘으면 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수로, 활성화 함수로 사용된다.

def relu(x):
    return np.maximum(0,x) #0과 x 중 큰 값을 return 


# # 3.3 다차원 배열의 계산
# ## 3.3.1 다차원 배열

A=np.array([1,2,3,4])
print(A)

np.ndim(A)

A.shape #1차원 배열도 shape의 return 값으로 tuple을 반환 -> 다른 다차원 배열의 shape과 형태를 통일하여 return 하기 위함

A.shape[0]

B=np.array([[1,2],[3,4],[5,6]])
print(B)

np.ndim(B)

B.shape

B.shape[1]

# ## 3.3.2 행렬의 곱

A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
A.shape

B.shape

np.dot(A,B)

A=np.array([[1,2,3],[4,5,6]])
A.shape

B=np.array([[1,2],[3,4],[5,6]])
B.shape

np.dot(A,B)

A=np.array([[1,2],[3,4],[5,6]])
A.shape

B=np.array([7,8])
B.shape #1차원 배열의 shape를 1*2 가 아닌 (2,)로 표현하는 것에 주의

np.dot(A,B)

# ## 3층의 신경망 구현하기
# ### 다차원 배열 사용

# +

X=np.array([1.0,0.5]) #입력 노드의 값
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) #weight
B1=np.array([0.1,0.2,0.3]) #bias

print(W1.shape)
# -

print(X.shape)

print(B1.shape)

A1=np.dot(X,W1)+B1

# +
Z1=sigmoid(A1)

print(A1)
print(Z1)
# -

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)


# +
def identity_function(x):
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y=identity_function(A3)
# -

print(Y)


# ## 3.4.3 구현 정리
#

# +
def init_network():
    network={} #dictionary
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    
    return network

def forward(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1) #활성화 함수 처리
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)
    
    return y

network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)
# -

# ## 3.5 출력층 설계하기
# ### 3.5.1 항등함수와 소프트맥스 함수 구현하기
# #### -classification 과 regression 이 대표적인 기계학습에서의 문제이다.
# #### -classificatioin(분류)에서는 주로 소프트맥스 함수를 출력층에서 활성화 함수로 사용한다.
# #### -regression(회귀)에서는 주로 항등 함수를 출력층에서 활성화 함수로 사용한다.

a=np.array([0.3,2.9,4.0])
exp_a=np.exp(a)
print(exp_a)

sum_exp_a=np.sum(exp_a)
print(sum_exp_a)

y=exp_a/sum_exp_a
print(y)

# ### 3.5.2 소프트맥스 함수 구현 시 주의점 (overflow 관점)

a=np.array([1010,1000,990])
np.exp(a)/np.sum(np.exp(a)) #overflow 발생

c=np.max(a)
print(c)

a-c

np.exp(a-c)/np.sum(np.exp(a-c))


