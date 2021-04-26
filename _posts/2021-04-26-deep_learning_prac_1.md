---
title: 딥러닝 공부. 각종 링크 저장용
categories: [study]
comments: true
---

## 공부할 때 참고하기 좋은 사이트

https://github.com/teddylee777/machine-learning


## RNN

rnn 은 이전의 출력과 현재의 입력을 입력으로 받아 새로운 출력을 만들어내는 네트워크이다.

이러한 네트워크가 여러개 이어붙여졌을 때,

gradient를 계산하는 과정에서 vanishing gradient 나 exploiding gradient 가 발생할 수 있다고 한다.

이를 극복하기 위해 만들어진 것이 LSTM 이다.

backpropgation through time 이라는 개념으로 알려져 있는 것인데,

https://m.blog.naver.com/infoefficien/221210061511


의 내용을 참고해보면 좋겠다.

## LSTM

lstm 은 long term 메모리와 hidden state 라는 개념이 있다.

일반적인 rnn 에 long term 메모리라는 것이 추가된 것이다.

https://www.youtube.com/watch?v=bX6GLbpw-A4&list=PLVNY1HnUlO24lnGmxdwTgfXkd4qhDbEkG&index=15

## GAN

네이버 D2 에서 있었던 강의이다.

https://www.youtube.com/watch?v=odpjk7_tGY0

GAN 에서 Generator 는 어떤 데이터의 확률분포를 학습해서, latent space 상의 임의의 값과 데이터를 매핑한다.

Discriminator 는 Generator 가 생성한 데이터와 실제 데이터를 구분하기 위해 학습하고,

Generator 는 Discriminator 를 속이기 위해 학습한다.

## Auto Encoder

https://www.youtube.com/watch?v=o_peo6U7IRM

## Transformer

Robust motion in-betweening 에서 embedding 이라는 개념을 말할 때, transformer 에 대해 논한다.

근래에 neural graphics 에서도 자주 쓰이는 모델인 듯 하니 익혀보면 좋을 듯 하다.

https://www.youtube.com/watch?v=AA621UofTUA
