---
title: 논문 리뷰. Human Motion Prediction vis Spatio-Temporal Inpainting
categories: [study]
comments: true
---

## 리뷰

이 논문에서는 motion prediction 을 위해 "시간축 x 위치정보(프레임)"을 하나의 텍스쳐처럼 생각해서 CNN을 통해 inpainting 하는 것을 생각하였다.

우선 각 프레임에서 입력으로는 각 조인트의 위치정보가 들어가게 되는데, 이 position 이 나열된 순서와 실제 공간상에서 joint와 joint 사이의 연결관계는 다르기 때문에 Encoding 을 해주게 된다.

즉, 각 프레임의 latent space 를 시간축으로 나열해놓고, inpainting 을 진행한다.

또한, Encoder - Inpainting (CNN U-Block) - Decoder 구조를 통해 inpainting 을 진행하고 난 결과 뒤에 3개의 discriminator 를 붙였다.

사용한 loss 는 reconstruction loss 와 limb distance loss (몇 가지 지정한 조인트와 조인트 사이 거리의 ground truth 와의 차이), bone length loss (아무래도 global position 을 출력으로 하기 때문에 필요하다.) 이고, 추가적으로 adversarial loss 를 사용했다.

limb distance loss 를 넣은 근거는 모션의 종류를 결정할 때, 특정 본과 본사이의 거리가 아주 유용한 지표가 된다는 것이다.

이전 논문(Robust Motion In-betweening)에서 reconstruction loss 는 mean 을 따라가지만, 세세한 모션을 재현하지 못해서 sliding 하는 듯한 경향성을 보이게 된다는 이야기가 있었는데, 본 논문에서도 adversarial loss 를 추가한 것을 보면 요즘의 트랜드가 아닌가 싶다. (RNN 에 의한 모션 생성 또한 blurriness 가 발생한다고 한다.)

이 논문에서는 기존의 L2 metric 이 long-term prediction 을 평가하기에 좋지 않다고 발하며, 새로운 metric 두 개를 제시하였는데, 둘 다 푸리에 트렌스폼을 적용한 cross entropy 나 kl divergence 이다.

최종적으로 본 논문의 결과물이 좋았는지 알기 힘들다. 다른 기술들과의 퍼포먼스 비교 자료도 없고, 순수하게 discriminator 가 붙음에 따라서 퍼포먼스가 어느 정도 향상되었는가를 따졌다. 또한 평가한 지표도 논문에서 제시한 metric 이었다.

## 익힌 용어

metric : 모델을 학습한 이후 그것이 잘 동작하는지 평가하기 위한 방법론. loss 는 낮아지는 동안 그것이 실제로 일을 얼마나 잘하는 지 지표가 될 수 없지만, metric 은 그것을 통해 사람이 보기에 맞는 종작을 하는지 평가할 수 있다.

reconstruction loss : output 으로 나온 데이터가 input 과 얼마나 유사한가를 따지는 loss. 보통 단순히 ground truth 와의 L1, L2 norm 으로 구하는 모양이다.

anthropomorphism : 아마도 인간을 따라하는 정도, 혹은 인간스러움. 이라는 뜻으로 보인다.

## 개인적인 생각

시간과 데이터를 2차원 데이터로 해석해서 inpainting 을 진행한다는 것은 꽤나 좋은 아이디어로 보인다.

하지만 결과물을 보면 ground-truth 와 차이가 크게 나는 것으로 보인다.

왜 그런지 그냥 생각나는 것들을 적어본다. 

우선 입력과 출력이 global 좌표였다는 점은, 요근래에 velocity-space 상에서 학습시 더 좋은 결과를 보였다는 경향과 다르다.

또한 inpainting 을 위해서 convolution 을 하게 되는데, convolution 이 충분히 long-term 의 움직임을 잘 포착하는 도구가 되어 주었을까? 하는 의문이 생긴다. RNN 의 좋은 대체제가 되어야 할 듯 한데, convolution 과정에서 최대 몇 프레임까지 멀리 볼 수 있을 지가 중요해보인다.

최근 state of the art inpainting 테크닉중 하나로 Convolutuin 과정에 mask 라는 개념이 추가된 것이 있는데, 이것을 활용했다면 더 좋은 퍼포먼스를 만들어냈을까?
