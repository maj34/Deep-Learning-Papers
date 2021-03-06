# SPPNet: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

**Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun**

# Abstract

현존하는 CNN은 **고정 크기의 입력 이미지를 요구**

이는 '인위적'이며 그 이미지나 임의의 크기로 변환된 부분 이미지에 대한 인식 정확도를 해침

→ 'spatial pyramid pooling'이라는 다른 방법을 제시

이미지의 크기에 상관없이 고정 길이의 대표값(representation)을 생성

pyramid pooling은 **object 변형에도 강건**함

**Contribution**

- CNN 기반의 이미지 classification을 개선하여 **ImageNet 2012**에 결과를 보여줌
- PASCAL VOC 2007 및 Caltech101 dataset에 대해서도 하나의 이미지 대표값으로 fine-tuning 없이도 최고의 결과를 보여줌
- object detection에서도 강력한데, 전체 이미지에서 feature map을 한번만 계산하고 detector를 학습하기 위해 임의의 region에 대해 고정 크기의 대표값을 생성함
- 테스트 이미지 처리시에 이 방법은 **R-CNN보다 20~102배 빠르고** 정확도는 비슷함
- ILSVRC 2014에서 **detection 부문 2위**, **classification 부문 3위**
    
    

# 1. Introduction

기존의 CNN 아키텍쳐들은 모두 **입력 이미지가 고정**되어야 했음 (ex. 224 x 224) 

→ 신경망을 통과시키기 위해서는 이미지를 **고정된 크기로 크롭**하거나 **비율을 조정**(warp)해야 함

하지만 이렇게 되면 물체의 일부분이 잘리거나, 본래의 생김새와 달라지는 문제점

***"입력 이미지의 크기나 비율에 관계 없이 CNN을 학습 시킬 수는 없을까?"***

![Untitled](images/Untitled.png)

Convolution 필터들은 사실 입력 이미지가 고정될 필요가 없음

sliding window 방식으로 작동하기 때문에, 입력 이미지의 크기나 비율에 관계 없이 작동함

입력 이미지 크기의 고정이 필요한 이유는 바로 컨볼루션 레이어들 다음에 이어지는 **fully connected layer**가 고정된 크기의 입력을 받기 때문

여기서 **Spatial Pyramid Pooling(SPP)**이 제안됨

***"입력 이미지의 크기에 관계 없이 Conv layer들을 통과시키고,                          FC layer 통과전에 피쳐 맵들을 동일한 크기로 조절해주는 pooling을 적용하자!"*** 

입력 이미지의 크기를 조절하지 않은 채로 컨볼루션을 진행하면 

1. **원본 이미지의 특징**을 고스란히 간직한 피쳐 맵을 얻을 수 있음
2. 사물의 **크기 변화에 더 견고**한 모델을 얻을 수 있음
3. Image Classification이나 Object Detection과 같은 여러 테스크들에 **일반적으로 적용**할 수 있음

<aside>
💡 **전체 알고리즘**

1. 먼저 전체 이미지를 미리 학습된 **CNN을 통과시켜 피쳐맵**을 추출함

2. Selective Search를 통해서 찾은 각각의 RoI들은 제 각기 크기와 비율이 다름 

    이에 SPP를 적용하여 **고정된 크기의 feature vector**를 추출함

3. 그 다음 **fully connected layer**들을 통과 시킴

4. 앞서 추출한 벡터로 각 이미지 클래스 별로 **binary SVM Classifier**를 학습시킴

5. 마찬가지로 앞서 추출한 벡터로 **bounding box regressor**를 학습시킴

</aside>

본 논문의 가장 핵심은 Spatial Pyramid Pooling을 통해서 **각기 크기가 다른 CNN 피쳐맵 인풋으로부터 고정된 크기의 feature vector를 뽑아내는 것**에 있음

# 2. Deep Networks with Spatial Pyramid Pooling

### 2.1. Convolutional Layers and Feature Maps

![(a) Pascal VOC 2007의 2개의 이미지
(b) conv5 의 특정 피처 맵
(c) 대응하는 필터의 응답이 가장 강한 receptive field](images/Untitled%201.png)

(a) Pascal VOC 2007의 2개의 이미지
(b) conv5 의 특정 피처 맵
(c) 대응하는 필터의 응답이 가장 강한 receptive field

convolution 층은 sliding filters를 사용하며, 그 출력은 입력과 동일한 aspect ratio를 가짐

필터는 의미 있는 content 에 의해서 active 하게 됨

ex. 55번째 필터(왼쪽 하단)는 원 모양으로, 66번째 필터(오른쪽 상단)는 ∧ 모양, 118번째 필터(오른쪽 하단)는 ∨ 모양으로 가장 활성화 됨

### 2.2. The Spatial Pyramid Pooling Layer

![SPP Layer가 있는 네트워크 구조](images/Untitled%202.png)

SPP Layer가 있는 네트워크 구조

1. 먼저, Conv Layer들을 거쳐서 추출된 **feature map을 인풋**으로 받음
2. 그리고 이를 미리 정해져 있는 영역으로 나누어 줌 
    
    (위의 경우, 미리 4x4, 2x2, 1x1 세 가지 영역을 제공하며, 각각을 하나의 **피라미드**라고 부름. 즉, 해당 예시에서는 **3개의 피라미드**를 설정한 것, **피라미드의 한 칸을 bin** 이라고 함)
    
    ex. 64 x 64 x 256 크기의 피쳐 맵이 들어온다고 했을 때, 4x4의 피라미드의 bin의 크기는 16x16
    
3. 이제 각 bin에서 가장 큰 값만 추출하는 **max pooling**을 수행하고, 그 결과를 쭉 이어붙여 줌 
4. 입력 피쳐맵 **채널 크기를 k**, **bin의 개수를 M**이라고 했을 때 SPP의 최종 아웃풋은 k*M 차원 벡터
    
    (위의 예시에서 k = 256, M = (16 + 4 + 1) = 21)
    

→ 입력 이미지의 크기와 상관없이 **미리 설정한 bin 개수와 CNN 채널 값으로 SPP의 출력이 결정됨**

→ 항상 동일한 크기의 결과를 리턴한다고 볼 수 있음 

실제 실험에서 저자들은 1x1, 2x2, 3x3, 6x6 총 4개의 피라미드, 50개의 bin으로 SPP를 적용

### 2.3. Training the Network

이론적으로 위의 네트워크는 **입력 이미지 크기에 관계없이** 표준 back-propagation을 사용해 훈련

but, 실제로는 cuda-convnet, Caffe와 같은 GPU 구현시 **고정 이미지 크기를 선호**함

다음은 SPP 동작을 유지하면서 이러한 GPU 구현의 이점을 살리는 학습 방법을 기술함

***Single-size training***

이전 작업처럼 224x224의 고정 크기로 crop된 이미지를 먼저 고려하면 crop은 data 증강을 위한 것

주어진 이미지 크기에 대해 먼저 SPP에 필요한 bin 사이즈를 미리 계산함

**SPP 예시**

- ROI feature - 13x13
- Spatial bin - 3x3 (pooling 연산을 통해 3x3 feature map을 얻겠다는 의미)
- 아래의  ROI feature에서 3x3 feature map을 얻기 위해서는 window size =5, stride = 4 로 설정
- 아래와 같이 설정된 window (작은 네모 박스)는 이동하면서 max pooling 연산을 적용

![SPP 적용 예시](images/Untitled%203.png)

SPP 적용 예시

![cuda-convnet 환경에서 구현된 3-level pyramid (3x3, 2x2, 1x1)](images/Untitled%204.png)

cuda-convnet 환경에서 구현된 3-level pyramid (3x3, 2x2, 1x1)

but, 보통은 1x1, 2x2, 4x4 spatial bin 사용

다양한 spatial bin을 가지고 있다는 의미에서 Spatial Pyramid 라고 함

위의 SPP를 통해 1x1, 2x2, 4x4 spatial bin 을 얻었다면 spatial bin을 모두 flatten하게 됨

그럼 총 16+4+1=21개의 feature가 만들어 지는 것!

21개의 고정된 feature들은 fc layer로 넘어가게 됨

***Multi-size training***

![single-size training vs. multi-size training](images/Untitled%205.png)

single-size training vs. multi-size training

다양한 크기의 서로 다른 input image size를 넣어 네트워크의 **수렴을 빠르게 하고 성능을 개선**함

![Untitled](images/Untitled%206.png)

**이미지를 6개의 스케일** s=(224, 256, 300, 360, 448, 560) 로 조절

각 스케일에 대해 전체 이미지에 대한 feature map을 계산

어떤 스케일에서든 224x224를 뷰 크기로 사용하여 이 뷰는 원래 다른 스케일의 이미지에 대해 상대적으로 다른 크기를 가지게 됨 

각 스케일에 18개의 뷰를 사용 (4 코너와 중앙, 각 변 중심에서 4개, 각각을 플립)

![Untitled](images/Untitled%207.png)

4가지 네트워크 구조를 활용하여 SPP-Net을 적용한 결과 개선되는 점을 실험

# 3. SPP-Net for Image Classification

### 3.1. Experiments on ImageNet 2012 Classification

### 3.1.1. Baseline Network Architectures

### 3.1.2. Multi-level Pooling Improves Accuracy

![성능 : multi-size trained > single-size trained > no SPP](images/Untitled%208.png)

성능 : multi-size trained > single-size trained > no SPP

4개의 level pyramid, 50개의 bin 

multi-size trained를 사용하면 parameter를 더 사용하면서 object의 변형에 robust 에러율 향상

![성능 : 전체 이미지 > 잘린 이미지](images/Untitled%209.png)

성능 : 전체 이미지 > 잘린 이미지

### 3.1.3. Multi-size Training Improves Accuracy

### 3.1.4. Full-image Representations Improve Accuracy

### 3.1.5. Multi-view Testing on Feature Maps

![Untitled](images/Untitled%2010.png)

ImageNet 2012 에서 최고 수준이었던 모델보다 에러율 향상

### 3.1.6. Summary and Results for ILSVRC 2014

![Untitled](images/Untitled%2011.png)

ILSVRC 2014에서 3위에 해당하는 좋은 결과를 보여줌

### 3.2. Experiments on VOC 2007 Classification

![Untitled](images/Untitled%2012.png)

![Untitled](images/Untitled%2013.png)

### 3.3. Experiments on Caltech 101

![Untitled](images/Untitled%2014.png)

# 4. SPP-Net for Object Detection

### 4.1. Detection Algorithm

Object Detection에 SPP를 적용할 수 있음

저자들은 R-CNN의 문제점을 지적하며 SPP를 이용한 더 효율적인 object detection을 제안 

R-CNN은 Selective Search로 찾은 2천개의 물체 영역을 모두 고정 크기로 조절한 다음, 미리 학습된 CNN 모델을 통과시켜 feature를 추출 → 속도가 엄청 느려짐

cf) SPPNet은 입력 이미지를 그대로 CNN에 통과시켜 피쳐 맵을 추출한 다음, 그 feature map에서 2천개의 물체 영역을 찾아 SPP를 적용하여 고정된 크기의 feature를 얻어냄

그리고 이를 FC와 SVM Classifier에 통과시킴

- **R-CNN vs. SPP-Net**
    
    ![R-CNN 네트워크 구조](images/Untitled%2015.png)
    
    R-CNN 네트워크 구조
    
    ![SPP-Net 네트워크 구조](images/Untitled%2016.png)
    
    SPP-Net 네트워크 구조
    
    ![R-CNN vs. SPP-net&Fast R-CNN](images/Untitled%2017.png)
    
    R-CNN vs. SPP-net&Fast R-CNN
    
    R-CNN에서는 입력 이미지에서부터 region proposal 방식을 이용해 candidate bounding box를 선별하고 모든 candidate bounding box에 대해서 CNN 작업을 함
    
    → 2000개의 candidate bounding box가 나오게 되면 2000번의 CNN 과정을 수행
    
    SPP-Net은 입력 이미지를 먼저 CNN 작업을 진행하고 다섯번째 conv layer에 도달한 feature map을 기반으로 region proposal 방식을 적용해 candidate bounding box를 선별
    
    → CNN 연산은 1번이 됨
    
    ![Untitled](images/Untitled%2018.png)
    
    ∴ R-CNN 2000번 → SPP-Net 1번의 CNN Operation 절감효과, 시간을 빠르게 단축
    
    ![Training & Test Time](images/Untitled%2019.png)
    
    Training & Test Time
    

### 4.2. Detection Results

![Untitled](images/Untitled%2020.png)

![Untitled](images/Untitled%2021.png)

Pascal VOC 2007 mAP 결과값 

scale을 변화시키며 SPP-Net를 적용한 결과

vs. R-CNN 을 이용해 fine-tuning과 bounding box regression을 이용한 결과 더 좋은 성능

### 4.3. Complexity and Running Time

![Untitled](images/Untitled%2022.png)

![Untitled](images/Untitled%2023.png)

### 4.4. Model Combination for Detection

### 4.5. ILSVRC 2014 Detection

![Untitled](images/Untitled%2024.png)

ILSVRC 2014에서 2위에 해당하는 좋은 결과를 보여줌

# 5. Conclusion

SPPNet은 기존 R-CNN이 모든 RoI에 대해서 CNN inference를 한다는 문제점을 획기적으로 개선

하지만 여전히 한계점이 있는데, 

1. end-to-end 방식이 아니라 **학습에 여러 단계**가 필요함 

    (fine-tuning, SVM training, Bounding Box Regression)

2. 여전히 최종 클래시피케이션은 binary SVM, Region Proposal은 **Selective Search**를 이용

3. **fine tuning 시**에 SPP를 거치기 이전의 Conv 레이어들을 학습 시키지 못함

    단지 그 뒤에 **Fully Connnected Layer만 학습**시킨다.

    → 저자들은 ***"for simplicity"***  라고만 설명함

# Appendix A

![Untitled](images/Untitled%2025.png)

# **Reference**

- **Object Detection 논문 흐름 및 리뷰**

[[Object Detection] 1. Object Detection 논문 흐름 및 리뷰](https://nuggy875.tistory.com/20)

- **GitHub 링크** (Papers with Code 기준)

[GitHub - yueruchen/sppnet-pytorch: A simple Spatial Pyramid Pooling layer which could be added in CNN](https://github.com/yueruchen/sppnet-pytorch)

- **논문 리뷰 - 유튜브**

[[Paper Review] Introduction to Object Detection Task : Overfeat, RCNN, SPPNet, FastRCNN](https://www.youtube.com/watch?v=SMEtbrqJ2YI)

[천우진 - Spatial pyramid pooling in deep convolutional networks for visual recognition](https://www.youtube.com/watch?v=i0lkmULXwe0)

- **논문 리뷰 - 블로그**

[6. SPP Net](https://89douner.tistory.com/89)

[갈아먹는 Object Detection [2] Spatial Pyramid Pooling Network](https://yeomko.tistory.com/14)

[[논문 리뷰] SPPnet (2014) 리뷰, Spatial Pyramid Pooling Network](https://deep-learning-study.tistory.com/445)

[SPPnet](https://blog.daum.net/sotongman/7)