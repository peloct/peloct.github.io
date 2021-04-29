---
title: 딥러닝 공부 2. Transformer
categories: [study]
comments: true
---

robust motion in-betweening 이라는 논문을 읽다가,

transformer 라는 기술의 positional encoding 이라는 것을 활용했다라는 것을 보고 잠깐 공부해보았다.

https://www.youtube.com/watch?v=AA621UofTUA

위의 동영상에서 학습하였다.

## NLP 에서 seq2seq 모델의 문제점

문장을 번역하는 인공지능을 딥러닝 기반으로 학습시킨다고 하자.

seq2seq 모델은 RNN 에 문장을 구성하는 모든 단어를 집어넣고, 그 끝에 최종적으로 나온 hidden state 를 이용해서

다시 또 다른 RNN 을 돌리며 번역된 문장을 만들어내는 기술이다.

그런데 이러한 방법론은 문장이 길어질 수록 고정된 hidden state 사이즈 안에

긴 문장에 대한 정보를 전부 집어넣게 되기에 점차 퍼포먼스가 떨어지게 된다는 단점이 있었다.

## Attention

이에 각 단어를 입력 인코더(RNN)에 집어넣으면서 만들어진 각 단계의 hidden state 를

활용하자는 아이디어가 나온다. 현재 디코더가 어떤 단어까지 만들었다고 했을 때,

입력단에서 생겨난 각 hidden state 에 대해서 모종의 방법으로 (어떤 방법을 쓰는지 종류도 다양하다.)

가장 중요하다고 여겨지는 hidden state 를 결정한다. 가장 중요한 hidden state에 가장 높은 가중치가

먹여지게 되는 형태인데, 각 hidden state 를 가중치에 따라 softmax 하면 새로운 hidden state 가 구성된다.

이것을 이용해서 새로운 단어를 결정하기 위한 로직이 돌게 된다.

## Transformer

기존 attention 방법론도 본질적으로 rnn 이기에 학습이 어렵다는 점은 변하지 않는다.

그렇기 때문에 rnn 을 완전히 제거하고, attention 방법론을 최대한 활용하는 기술이 나왔다.

여기서 내가 관심을 가진 것은 positional encoding 이라는 개념이다.

각 단어에 대해서 attention 을 결정해야 하는데, 문장이라는 것은 단어가 어떤 순서로 오는지가

의미 형성에 큰 역할을 한다. 따라서 각 단어가 문장에서 몇 번째로 오는 지에 대한 정보가 필요하게 되는데,

이를 positional encoding 을 통해서 해결한다.

각 단어의 embedding vector 에 positional encoding 이라는 벡터를 더해주는데,

position embedding 은 사인과 코사인으로 구성된 어떤 함수이다.

embedding vector 의 $$i$$ 번째 요소에 더해질 값에 대해서,

$$i$$ 가 짝수인 경우,

$$ f(p, i) = sin({p}/10000^{i/d_{model} }) $$

$$i$$ 가 홀수인 경우,

$$ f(p, i) = cos({p}/10000^{i/d_{model} }) $$

$$d_{model}$$ 은 embedding vector 의 사이즈, $$p$$ 는 위치값이다.

위와 같은 벡터를 각 단어의 embedding vector에 더해주는 것으로,

위치정보를 네트워크에 넘겨준다.
