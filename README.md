# CNN-with-EM-Algorithm-Loss

> **목적**: 기존의 **Cross Entropy** 손실만으로는 부족한 표현 학습 한계를 극복하기 위해, **K‑means 기반 EM(Expectation–Maximization) Loss**를 결합하여 **클래스 간 분리를 극대화**하고 **클래스 내부 응집도를 향상**시키는 CNN 학습 파이프라인을 제공합니다.

---

## 📚 프로젝트 개요

본 레포지토리는 두 가지 학습 시나리오를 비교·분석합니다.

1. **Baseline** ‑ 순수 Cross Entropy(CE) Loss 로 CNN(VGG19+BatchNorm) 학습
2. **Proposed** ‑ Cross Entropy + **EM‑Algorithm Loss**(K‑means clustering on softmax probablitiy.)

학습 완료 후 **t‑SNE**로 투영된 특징 공간과 **Grad‑CAM** 시각화를 통해 두 모델의 표현력을 정량·정성적으로 평가합니다. 그 결과, EM‑Loss를 적용한 모델이 **더 조밀한 클래스 매니폴드**와  **정확한 활성 위치(=Heatmap)** 를 보여 줌을 확인했습니다.

---
## ✨ Result ) 단순 Cross Entropy Loss 와 비교.

*Grad CAM 비교*
---
![image](https://github.com/user-attachments/assets/39d4bbc8-eaff-4b93-9186-b2eaa5df47c8)

---

*Manifold Space 비교*
---
![image](https://github.com/user-attachments/assets/f1fd6bef-e460-493e-bff5-4f807109dd6d)


## 🧮 EM‑Algorithm Loss 수식

![image](https://github.com/user-attachments/assets/5b11539b-0ca3-494c-8d79-4b293aedabe7)

*변수 설명*
---
| 기호 | 의미 |
|------|------|
| **p<sub>i</sub>** | i‑번째 샘플의 softmax 확률 벡터. 각 클래스에 속할 확률 분포를 나타냅니다. |
| **μ<sub>k</sub>** | k‑번째 클러스터(또는 클래스) 중심. 동일 클래스 확률 벡터들의 평균으로 계산됩니다. |
| **K** | 현재 배치에 존재하는 클래스(클러스터) 수. |
| **λ** | Inter‑cluster 항의 가중치. 클래스 간 분산을 얼마나 강조할지 조절하는 스칼라 하이퍼파라미터. |
| **Intra** | 동일 클래스 내부에서의 응집도를 측정하는 항. 값이 작을수록 같은 클래스 샘플들이 가깝게 모여 있음을 의미합니다. |
| **Inter** | 서로 다른 클래스 중심 간의 분리를 측정하는 항. 값이 클수록 클래스 간 간격이 넓어집니다. |

## ⚙️ 로스 함수 작동 원리

배치에 포함된 확률 벡터들을 **K‑means**로 군집화한 뒤,

1. **Intra 항**을 통해 동일 클래스 샘플 간 거리를 최소화하여 **응집도**를 높이고,
2. **Inter 항**을 통해 다른 클래스 중심 간 거리를 λ배만큼 크게 만들어 **분리도**를 확보합니다.

최종 손실은 전통적인 교차 엔트로피 손실 **L<sub>CE</sub>** 에 EM‑Loss를 가중치 **α**로 더한

**L = L<sub>CE</sub> + α · L<sub>EM</sub>**

형태로 정의되어, 정확도와 특징 공간 분별력을 동시에 향상시키도록 설계되었습니다.


---
