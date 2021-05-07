---
title: 딥러닝 공부. Variational Autoencoder
categories: [study]
comments: true
---

## Variational Autoencoder

다음의 블로그를 보고 익힌 것, 느낀 것을 적고자 한다.

https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

encoder에 대해서는 적당히 latent space로 데이터를 변환하고, decoder는 그것을 다시 데이터로 변환해주는 것이다. 라고 이해하고 있었다.

만약 Network를 구성하길 Encoder - Decoder 로 만들어서, 입력과 출력이 같도록 학습을 해주게 되면,

이때 Encoder는 알아서 자신의 뒤에 붙은 network가 원하는 출력을 할 수 있도록 데이터를 변형하는 법을 학습한다.

그래서 Autoencoder라 부르는 모양이다.

하지만 Generative Model 에서 단순히 autoencoder를 이용해 데이터 - latent space 매핑을 구성하게 되면, 단점이 있다고 한다.

latent space안에 유사한 데이터들끼리 클러스터를 구성하며 공간상에 옹기종기 모이게 될텐데,

network는 학습하는 과정속에서 decoder가 더 높은 정확도로 원본의 데이터를 만들어낼 수 있게끔 하기 위해,

회색영역을 없애나간다. 즉, 클러스터끼리 겹치지 않고 점점 구분되어가는 것이다.

이때 latent space 를 보면 클러스터가 전체 공간을 sparse하게 차지하게 되는데, 만약 latent space 에서 임의의 점을 찍었을 때,

그 점이 어떤 클러스터에도 들어가지 않게 되면 decoder는 해당 점에 대해서 우리가 원하는 출력을 만들어내지 못할 것이다.

이를 해결해서, latent space 상에 클러스터들을 원점을 기준으로 최대한 모이게 하고, 클러스터와 클러스터 사이에

빈공간을 제거하고 싶을텐데, Variational Autoencoder가 이를 해결한다.

Encoder 가 latent space 를 출력할 때, 피쳐의 평균과 variation 을 출력하도록 하고,

이 평균과 variation의 사이즈에 대해서 각각 0, 1이 되도록 regularization을 해주는 것이다.

이렇게 하면 클러스터들은 최대한 구의 형태를 갖추면서 원점으로 이동하고자 하는데, 생성된 데이터의 reconstruction loss에 의해

서로의 공간이 겹치지 않게 된다.

이렇게 만들어진 latent space 는, 그 안에 유효하지 않은 latent space 가 없기 때문에 interpolation 과 random sampling 이 가능하게 된다.

상당히 흥미로운 개념이다.

latent space 을 바라보는 시각이 확장되었다.
