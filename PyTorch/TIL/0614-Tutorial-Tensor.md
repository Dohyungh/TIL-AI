# Tutorial

**1. Tensor**  
2. Dataset, DataLoader  
3. Transform  
4. nn-model  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  

## Tensor
- 배열, 행렬과 유사 (특히 Numpy의 ndarray)
- 모델의 입력, 출력과 그 encoding에 사용 
    - `encoding` : 이미지, 오디오 -> nn에 전달 가능한 데이터로 변환
- GPU를 비롯한 Accelerator 에서 사용가능한 ndarray
- 자동미분(Autograd(Pytorch의 자동미분))에 최적화
    - Computational Graph
    - Gradient Tracking
    

