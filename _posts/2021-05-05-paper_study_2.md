---
title: 논문 리뷰. Action-Agnostic Human Pose Forecasting
categories: [study]
comments: true
---

## 논문 리뷰

본 논문에서는 본 모델이 운동할 때, 이후의 운동을 예측하는 주제를 다룬다.

이를 motion prediction 혹은 motion forecasting 이라고 부르는데,

기존의 연구에서는 언제나 short term prediction 과 long term prediction 사이에 트레이드 오프가 있었다고 한다.

요컨대 근미래에 대한 예측(~400ms)을 잘하는 알고리즘은 먼미래에 대한 예측(400~1000ms)을 잘 못하고

만약 먼미래에 대한 예측을 잘하는 알고리즘은 근미래에 대한 예측을 잘 못한다는 것이다.

논문에서는 서로 다른 time-scale에서 동작하는 LSTM 모듈을 여러개 (구현에서는 2개)를 붙이는 것으로

short term - long term 트레이드 오프가 없는 알고리즘을 만들 수 있었다고 한다.

또한 기존의 알고리즘들은 걷기면 걷기, 담배피기면 담배피기, 회의면 회의. 무언가 동작을 학습시킬 때,

특정 동작들만 모아서 학습시켜야 하는 문제가 있었다. (여러 종류의 모션을 동시에 학습시킬 수 x)

하지만 이 논문의 알고리즘은 그러한 분류 없이 한꺼번에 학습시킬 수 있다는 강점이 있다고 한다.

또한 velocity space 상에서의 학습을 통해서 (인풋, 아웃풋 모두 velocity 정보를 사용)

더 높은 퍼포먼스를 이끌어낼 수 있었다고 한다.

즉, 이 논문은 기존의 연구들과 다음에서 차이가 있다.

- 독자적인 LSTM 모듈 연결 구조 : TP-RNN
- Velocity Space 상에서의 학습

#### 왜 velocity space 인가?

저자는 velocity space 상에서의 학습이 좋은 이유에 대해서, input, output 의 scale 이 같기 때문에

뉴럴 네트웤이 더 쉽게 학습을 할 수 있다. 라고 주장한다.

또한 human pose 는 time-step 에 따라서 크게 변화하지 않게 때문에, pose 대신 velocity 를 쓰는 것이

prediction power 를 증가시킨다고 주장한다.

#### 왜 TP-RNN 인가?

저자가 Multi-scale RNN 구조가 필요하다고 생각하는 이유는, 인간 몸의 운동이 hierarchical, multi-scale 하기 때문이라고 한다.

예를 들어 걷기를 생각해보았을 때, 몸체의 운동이 손과 같은 body part에 영향을 준다. 요컨대 hierarchical 하다.

또한 몸통은 천천히 움직이는 반면, 손과 발은 빠른 주기로 반복한다. (multi-scale)

저자는 TP-RNN이 다양한 모션이 갖는 암묵적인 hierarchical, multi-scale dependancy를 학습하는 것으로,

별다른 모션 시스널(이러한 모션을 예측해보라라는 지시) 없이도, 다양한 모션을 학습하는 것이 가능했다.

라고 해석한다.

## 개인적인 생각들

Robust-Motion In-betweening 에서는 단일 LSTM 모듈을 이용해서 In-betweening 을 했다.

해당 논문의 저자는 본래 TP-RNN 구조를 사용하려고 했으나, TP-RNN 을 사용하나, 사용하지 않나 큰 퍼포먼스 상에

차이를 보이진 않았다고 한다.

그 원인은 명확하지 않다고 하며, encoder 네트워크가 들어감에 따라 TP-RNN 구조를 불필요하게 한 것은 아닌가

하고 해석하였다.

TP-RNN 이 성공적이었던 까닭은 (적어도 Action-Agnostic ~~ 논문의 저자의 주장으론) 여태까지 취해온 모션의

hierarchical, multi-scale 한 구조를 TP-RNN 이 캡쳐할 수 있었기 때문인 것인데,

만약 이것이 현재의 상태를 가지고 "지금 내가 무슨 행동을 하고 있었는지"를 추측하는 능력이 강해진 것이라고 해석할 수 있다면,

Robust-Motion In-betweening 에서 TP-RNN 이 필요없었던 이유를 어느 정도 설명할 수 있지 않을까 싶다.

Robust-Motion In-betweening 에서는 target 프레임에 대한 정보를 제공해주고 있는데,

네트워크가 이 target 프레임 정보를 토대로 지금 내가 무슨 행동을 하고 있는지 알아냈다는 해석이다.
