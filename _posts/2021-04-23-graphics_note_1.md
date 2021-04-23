---
title: 그래픽스 공부1
categories: [study]
comments: true
---

## pixel value 와 light intensity

단적으로 사람은 빛의 세기 (밝고 어두운 정도)의 비선형적인 변화(exponential한 변화)를 선형적으로 느끼는 듯하다.

예를 들어, 빛의 intensity가 1에서 2로 된 것과 2에서 4로 된 것을 같은 정도의 변화라고 느낀다.

인간이 느낄 수 있는 최소 변화 비율은 1.02 만큼의 변화라고 하는데

어떤 display가 출력하는 최소 intensity가 $I_{min}$, 최대 intensity가 $I_{max}$ 이면,

$$I_{min} * (1.02)^n = I_{max}$$

의 방정식을 푸는 것으로써, 해당 display에서 부드러운 그라데이션을 형성하기 위해 총 몇 번의 step을 거쳐야 하는지 알 수 있다.

또한 이 그라데이션은 사람이 인지하기에 선형적으로 증가하는 것처럼 보일 것이다.

여기서 $I_{max}\over I_{min}$ 을 dynamic range라고 하는데,

dynamic range가 클 수록, 부드러운 그라데이션을 형성하기 위해 필요한 $n$의 사이즈가 커진다.

그리고 이는 pixel의 밝기를 저장하기 위해서 요구되는 bit의 수가 증가한다는 것을 의미한다.

$$2^{bit number} \ge \log_{1.02} {I_{max}\over I_{min}}$$

일반적인 모니터에서는 dynamic range가 100 이라고 하는데 위의 식을 풀어보면,


```python
import numpy as np

a = np.log(100.0) / np.log(1.02)
min_bit_number = np.log(a) / np.log(2.0)
min_bit_number
```




    7.861418811173846



최소한 8bit가 필요하다는 것을 알 수 있다.

그리고 여기에서 알 수 있는 것은 pixel value는 실제 빛의 밝기가 아닌 인지적으로 선형적인 밝기를 나타내는 값임을 알 수 있다.

graphic library에서 pixel value를 넘겨주면, 이것을 display가 표현 가능한 실제 출력으로 변환하여 표시한다고 한다.


```python
import matplotlib.pyplot as plt

i_min = 1.0
i_max = 100.0

n = np.log(i_max / i_min) / np.log(1.02)

x = np.array([(i / 999.0) for i in range(1000)])
y = i_min * (1.02 ** (n * x))

plt.plot(x, y)
plt.show()
```


    
![png](/assets/img/graphics_note_1_files/graphics_note_1_4_0.png)
    


이상적으로는 대강 위와 같은 매핑으로 출력한다고 한다.(가로: normalized pixel value, 세로: intensity)

그렇기 때문에 만약 픽셀에 저장한 값이 물리적인 빛의 intensity라면,

모니터에 해당 intensity가 표시되도록 보정해줄 필요가 있다.

이를 gamma correction이라고 한다.
