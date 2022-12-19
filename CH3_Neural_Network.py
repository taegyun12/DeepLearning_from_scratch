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


