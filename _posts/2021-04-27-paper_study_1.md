---
title: 논문 리뷰. Robust Motion In-betweening
categories: [study]
comments: true
---

## 논문 리뷰

본 논문에서는 motion prediction 기술에 타겟 프레임을 지정하는 것(conditioning)을 통해서 transition generation 을 달성하고자 한다.

우선 타겟 프레임까지 남은 시간 (time to arrival, 이하 tta)을 네트워크에 넘겨주어야 하는 이유는 이 정보가 없을 때,

모션이 satalling 하거나 teleporting 하는 경향을 보이기 때문이다.

그 이유에 대한 내용은 논문에는 나와있지 않으나, 아무래도 tta 를 네트워크가 알지 못하면

해당 목적을 달성하기 위해 어떻게 모션을 만들어야 할지 네트워크가 판단하지 못하기 때문이 아닐까 싶다.

문제는 해당정보를 네트워크에 넘겨주는 방법론인데,

이 논문은 그 방법론에 있어서 NLP 에 높은 퍼포먼스를 보이고 있는 Transformer 기술을 활용했다는 점에서 차이를 보인다.

단순히 latent space 에 tta (단일 scalar) 를 붙여(concat)주는 방법론은

네트워크가 해당 정보를 무시하는 방향성으로 학습하기 쉬우며,

transformer 기술을 이용해 embedding modifier 를 latent space 에 더해주는 방법을 사용할 시에는

그 정보를 무시하기 힘들기 때문에 더 나은 결과가 나오는 것이 아닐까라고 저자는 추측한다.

또한 tta 뿐만 아니라 target noise라 하여, tta 가 클 수록 커지는 nosing factor 를

latent space 에 마찬가지로 더해주는데, 이를 통해서 네트워크를 noise 에 대해 robust 하게 하면서

동시에 transition generation 에 variation 을 줄 수 있게 된다.

이 또한 마찬가지로 latent space 자체에 더한 것이기 때문에 네트워크가 무시하기 힘들다고 본다.

## 관련 기술 혹은 용어에 대한 조사

#### motion prediction 과 NLP 기술의 연관성

ERD 는 Encoder - Recurrent - Decoder 의 줄임말인데, motion prediction 에서 최근에 쓰이는 구조들의 기반이 되는

모델이다. 이 모델의 해석방법에 대해서는 Terry 라는 분이 On Human Motion Prediction using RNNs 라는 논문을

리뷰하는 동영상에서 알 수 있었는데, ERD 모델은 본질적으로 NLP 에서 알려진 seq2seq 기술의 모션 버전이라고 한다.

모션은 자연어에 비해 real value 이면서, 데이터 사이에 의존성이 존재하고, 차원수가 매우 높다는 점이

어렵다고 한다. 이러한 모션을 latent space 로 매핑하여 마치 하나의 단어처럼 생각하도록 하는 것이 Encoder 의 역할이다.

요컨대 Encoder 와 Decoder 를 통해 모션을 latent space 로 오가게 하는 것으로,

motion prediction 문제를 seq2seq 으로 해결하고자 하는 것이다.

#### noise 를 넣어 학습하는 것

motion prediction 알고리즘들은 generation 이 오랫동안 지속됨에 따라 error 가 누적되는 특징을 보인다.

error 가 누적됨에 따라 학습되지 않은 region 에 들어가게 될 가능성이 크기 때문이다.

이를 해소하기 위해 학습과정에서 noise 를 집어넣게 되는데, noise 를 넣게 되면 어느 정도 error 에 대해서

robust 하게 학습하게 된다고 한다. 문제는 noise 튜닝이 어렵다는 점이다.

generation 이 지속됨에 따라 error 가 점점 커지는 것을 따라하기 위해 noise 또한 점점 키우는 형태가 된다.

#### On Human Motion Prediction using RNN

이 단락에서 말하는 "이 논문"이란 "On Human Motion Prediction using RNN"을 말한다.

이 논문에서는 기존 motion prediction 알고리즘이 몇 가지 문제점들을 해소하는 방법론을 제시했다.

- 문제1. 첫 프레임의 예측에서 순간이동 같은 점핑이 발생

우선 기존의 motion prediction 알고리즘은 그 output 으로 position 이나 quaternion 을 반환하였는데,

이 논문에서는 output 으로 속도를 반환하도록 했다. 이를 residual architecture 라고 부르는 모양인데,

왜 이것이 residual architecture 라고 불리는 지는 모르겠다.

어쨌든 이 방법론을 통해서 점핑을 해소할 수 있었다고 한다. output 의 값이 작기 때문이다.

- 문제2. noise tuning 의 어려움

기존의 연구들은 ground truth 한 입력을 주고 그에 대한 출력을 이용해서 학습시켰다고 한다.

하지만 이 논문에서는 ground truth 데이터 대신에, 이전에 예측한 데이터를 넣어서 학습시켰다고 한다.

자연스럽게 에러가 누적되는 효과가 생겨서 인지는 모르겠지만, 더 높은 퍼포먼스를 보였다고 한다.

#### CGAN 과의 연관성

CGAN 은 랜덤하게 샘플링된 latent space 에 여러가지 컨트롤 벡터를 concat한 이후 이를 generator 에 넣는 방법론이다.

본 논문도 latent space 에 tta 와 z_target 을 concat 하는 방법론에 대해서 논의를 했는데,

이것이 마치 CGAN 의 접근법과 유사하다고 평가하였다.

다만 앞서 이야기 했듯이, 현재 풀고자 하는 문제의 경우 condition 이 매우 informative 한 나머지

이렇게 concat 된 정보가 무시되는 경향이 있다고 한다.

#### robustness 의 의미

여러 논문에서 robust 하다라는 표현을 자주 보았는데, robust 하다는 것은

데이터에 에러가 있어도 안정적이거나, test, train 상에서 발생하는 error 가 일정함을 의미한다고 한다.
