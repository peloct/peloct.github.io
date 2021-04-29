---
title: 딥러닝 공부 1. 미분 및 각종 레이어 컴포넌트
categories: [study]
comments: true
---

## Gradient Descent

gradient descent 방법론은 어떤 함수의 값을 최소화하는 것을 목적으로 하며,

이 함수는 $$f(x; p)$$ 와 같이 표현된다. 여기서 $$p$$ 는 파라미터인데,

gradient descent 는 주어진 입력에 대해서 $$p$$ 가 함수 값에 미치는 영향 $${df} \over {dp}$$ 를 계산하고,

이를 이용해서 $$p$$ 를 변경하는 것으로 해당 목적을 이루어낸다.

$$ p \leftarrow p - \alpha { {df} \over {dp} } $$

위의 식을 계속해서 반복 적용하는 것이 gradient descent 방법론이라 할 수 있다. ($$\alpha$$ 는 learning rate)

## Deep Learning

deep neural network 는 비선형 함수를 만들어내는 방법론이다.

그 안에는 수없이 많은 파라미터들이 있는데,

gradient descent 따위의 optimization 기술들을 이용해서 이 함수를 최소화하는 파라미터들을 찾아내는 것이

deep learning 이라는 과정이다.

따라서 network 를 구성하는 다양한 컴포넌트들과 그들의 파라미터,

그리고 그 파라미터의 미분을 구하는 방법을 아는 것이 중요하다.

## 파라미터 미분구하는 요령

#### 수학 모델 표현

각 컴포넌트에 대해 파라미터의 미분을 구할 때, 가장 먼저 해야하는 것은 입력과 출력 데이터를 행렬, 벡터 등으로 표현하는 것이다.

이후 컴포넌트의 수학적 표현을 떠올리며 수식으로 이를 적어야 한다.

#### 구해야하는 미분은 $${dy} \over {dp}$$ 와 $${dy} \over {dx}$$

그 다음은 출력 $$y$$ 에 대해서 $${df} \over {dy}$$ 가 이미 구해져서 학습 알고리즘으로부터 이미 제공받았다고 가정하는 것이다.

딥러닝의 back propagation 은 chain rule 에 기반하며, 가장 나중에 적용되는 컴포넌트부터 미분이 구해지기 때문에

$${df} \over {dy}$$ 는 반드시 구해질 수 밖에 없다.

$${df} \over {dy}$$ 를 얻었기 때문에 $${dy} \over {dp}$$ 와 $${dy} \over {dx}$$ 를 구하기만 하면,

체인룰에 의해 $${df} \over {dp}$$ 와 $${df} \over {dx}$$ 모두 구할 수 있다는 사실을 떠올려야 한다.

전자는 파라미터를 업데이트하는데에, 후자는 이후 컴포넌트의 출력에 대한 미분을 제공하는데에 사용된다.

#### $$p$$, $$x$$의 차원은 $${df} \over {dp}$$ 와 $${df} \over {dx}$$ 의 차원과 같다.

이것은 꽤나 중요한 사실인데, 실컷 계산한 다음 만약 위의 규칙이 지켜지지 않으면, 어딘가 잘못 계산했다는 증거이다.

또한 이 규칙이 성립하기 위해선 $$f$$ (딥러닝이므로 cost function) 이 단 하나의 실수를 내뱉는 함수여야한다.

딥러닝의 경우 보통 다음과 같은 꼴의 cost function 을 사용한다.

$$ f = {1 \over m} \sum_i^m { L(N(x^{(i)}), y^{(i)}) } $$

여기서 $$m$$ 은 데이터 수, $$L$$ 은 loss function, $$N$$ 은 network 이다.

#### 미분은 원소 단위로 한다.

식이 행렬과 벡터 연산으로 구성되는 경우가 많다보니, 행렬과 벡터 미분을 사용하고 싶은 충동이 들지만,

그 방법론에 아주 능숙한 것이 아니라면 다음과 같이 원소단위로 구하는 것이 아주 편하다.

$${ {df} \over {dv_i} } = \sum_j { {df} \over {dy_j} } { {dy_j} \over {dv_i} } $$

이는 $$v$$는 미분을 구하고 싶은 어떤 대상이다. 파라미터 $$p$$일 수도 있고, 입력 $$x$$일 수도 있다.

또한 위의 식은 $$v$$가 벡터라는 가정을 넣었기에 index가 $$i$$ 밖에 없지만, 만약 행렬이라면 $$(i, j)$$ 처럼

여러 index가 요구될 수 있다.

## Linear Layer

말그대로 선형 레이어로, 주어진 입력에 선형 연산을 수행한다.

모든 선형 연산은 매트릭스 곱으로 표현이 가능하다는 사실은 선형대수에서 알 수 있다.

따라서 딥러닝에서 말하는 선형 레이어는 거대한 매트릭스에 입력을 곱하고, bias 라는 벡터를 더하는 레이어이다.

사실 bias 를 더하는 순간 이는 affine 변환이기에 비선형이지만,

각 입력의 끝에 1을 붙이고 매트릭스의 끝에 bias 벡터를 concat 하는 것으로 쉽게 선형 연산으로 변경할 수 있다.

$$ y = W x + b $$

input feature size 를 $$a$$, output feature size 를 $$b$$, batch size 를 $$m$$ 이라고 하면

$$y$$ 는 $$(b, m)$$ 차원 행렬이고, $$x$$ 는 $$(a, m)$$ 차원 행렬,

$$W$$ 는 $$(b, a)$$ 차원 행렬, $$b$$ 는 $$(b, 1)$$ 차원 행렬(혹은 $$b$$ 차원 벡터) 이다.

$${df} \over {dy}$$ 가 주어졌다는 가정하에 $$W$$ 의 미분을 구해보자.

$$ { {df} \over {dW_{i,j} } } = \sum_d \sum_c { {df} \over {dy_{c, d} } } { { dy_{c, d} } \over { dW_{i,j} } } $$

여기서 $$W$$ 의 $$i$$ 행 원소들은 $$y$$ 의 $$i$$ 행 원소들만을 결정할 수 있으므로,

$$ { {df} \over {dW_{i,j} } } = \sum_d { {df} \over {dy_{i, d} } } { { dy_{i, d} } \over { dW_{i,j} } } $$

이때

$$ y_{i, d} = \sum_k { W_{i, k} x_{k, d} } + b_i $$

이므로,

$$ { { dy_{i, d} } \over { dW_{i,j} } } = x_{j, d} = x^T_{d, j} $$

이다. 따라서

$$ { {df} \over {dW_{i,j} } } = \sum_d { {df} \over {dy_{i, d} } } x^T_{d, j} $$

여기서 $$  \sum_d { {df} \over {dy_{i, d} } } x^T_{d, j} $$ 은 $$ {df} \over {dy} $$ 와 $$ x^T $$ 를 곱해서 얻은

행렬의 $$i$$ 행 $$j$$ 열 원소를 말한다. 따라서

$$ { {df} \over {dW} } = { {df} \over {dy} } x^T $$

임을 알 수 있다. 위 식의 각 행렬의 차원수를 따져보면 잘 맞아 떨어짐을 알 수 있다.

비슷한 원리로 다른 파라미터, 입력에 대한 미분을 구할 수 있다.

## Activation Function

linear layer 만으로는 비선형성을 만들어낼 수 없기에, activation function 이라 불리는 함수들을 이용해서

linear layer 의 출력값을 변환해준다.

$$ y = f(Wx + b) $$

activation function 에는 여러 종류가 있는데, 각각의 수식과 특징을 간략하게 적어본다.

#### Sigmoid

$$y = \sigma(x) = { {1} \over {1 + e^{-x} } }$$

이 함수는 0부터 1사이의 값을 갖기 때문에 확률을 출력으로 할 때 요긴하게 사용된다.

zero centered 하지 않으면서, 동시에 항상 양의 부호를 갖기 때문에

sigmoid 의 출력값을 다른 linear layer의 입력으로 사용했을 때, 비효율적인 파라미터 업데이트를 유발할 수 있다.

$$ y = W\sigma(x) + b $$

여기서  $$ y $$ 가 단일 스칼라 값이라고 하자.

$$ { {df} \over {dW} } = { {df} \over {dy} } { {dy} \over {dW} } $$

여기서 $$ { {dy} \over {dW} } $$ 가 $$ \sigma(x) $$ 인데,

$$ \sigma(x) $$ 은 전부 같은 부호를 갖고 있고, $$ {df} \over {dy} $$ 는 스칼라 값이므로,

$$ { {df} \over {dW} } $$ 가 전부 같은 부호를 갖게 됨을 알 수 있다.

이로 인해 parameter space 상에서 파라미터가 지그재그로 업데이트 되게 되는데, 이를 zig-zagging 이라고 부른다.

또한 크거나 작은 입력에 대해서 그래디언트가 죽는다는 문제점이 있다.

#### Tanh

$$y = tanh(x) = { {e^{x} - e^{-x} } \over {e^{x} + e^{-x} } }$$

sigmoid 와 형태가 같지만, zero centered 하다.

-1 부터 1 사이의 값을 반환하고, 0에 가까운 곳에서 선형성을 갖는다.

sigmoid 와 마찬가지로 크거나 작은 입력에 대해서 그래디언트가 죽는다.

sigmoid 와 비교해 보았을 때, 거의 대부분 더 나은 퍼포먼스를 보인다고 한다.

#### ReLU

$$y = ReLU(x) = max(0, x)$$

zero centered 하지 않지만, 계산이 빠르고 sigmoid 보다도 6배 빠른 수렴속도를 보인다고 한다.

ReLU 를 사용했을 때, bias 가 음수로 매우 큰 경우 dying relu 라는 현상이 발생할 수 있다고 한다.

이는 해당 relu 가 계속해서 deactivate 되고, 그로 인해서 graient 가 지속적으로 0 이 되어

학습이 되지 않게 되는 현상이다.

이를 피하기 위해 초기에 weight initialization 단계에서 알맞은 weight 를 주는 것이 중요하다고 한다.

## Batch Normalization

보통 학습을 수행할 때, Batch 라는 단위로 전체 데이터 셋의 일부를 샘플링하여 네트워크를 학습시킨다.

이렇게 batch 단위로 학습시키다가, 전체 데이터 셋을 한 번 훑은 것을 1epoch 이라고 한다.

batch normalization 유닛은 말 그대로, 입력받은 피쳐 벡터가 batch size 만큼 있을 때,

이 벡터들의 평균과 분산을 구해서 normalization 해주는 것이다.

$$ \mu = { 1 \over m } \sum_i^m x^{(i)} $$

$$ \sigma^2 = { 1 \over m } \sum_i^m (x^{(i)} - \mu)^2 $$

$$ \hat{y}^{(i)} = { {x^{(i)} - \mu} \over { \sqrt { \sigma^2 + \epsilon } } } $$

$$ y^{(i)} = \gamma * \hat{y}^{(i)} + \beta $$

여기서 $$x^{(i)}$$ 는 입력받은 (배치 내) i번째 피쳐 벡터의 각 성분을 의미한다. 요컨대 단일 scalar 값이다.

batch normalization 의 경우 미분하기가 상당히 복잡하지만, 위에 적힌 요소단위 미분으로 천천히 접근하면 구할 수 있다.

https://kevinzakka.github.io/2016/09/14/batch_normalization/

위의 사이트를 참고해보아도 좋다.
