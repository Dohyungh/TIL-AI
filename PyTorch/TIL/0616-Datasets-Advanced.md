# Datasets-Advanced

1. Tensor  
2. Dataset,DataLoader  
   1. **`torch.utils.data`**
3. Transform  
4. nn-model  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---
> 핵심은 `torch.utils.data.DataLoader` 클래스이다. 그 생성자의 `arg`들을 하나씩 살펴보자. 

```
DataLoader(
      # 1
         dataset,  

      # 2
         batch_size=1,  
         shuffle=False,  
         sampler=None,  
         batch_sampler=None,   
         drop_last=False,
   
      # 3
         num_workers=0,   
         worker_init_fn=None, *,  
         collate_fn=None,  

      # 4
         pin_memory=False,   
         timeout=0,   
         prefetch_factor=2,   
         persistent_workers=False
         
         )   
```

## Dataset Style(Type)

### map-style
- `__getitem__()`
- `__len__()`
- 두 protocol을 구현해야 한다.
- idx를 통해 data와 label에 접근 가능하다.
- 즉, random access가 가능하다.
- tensorflow 에는 없다.
- 웬만하면 이걸 쓰면 된다.


### iterable-style
- `__iter__()`
- 하나의 protocol만 구현하면 된다.
- `next()` 메서드로만 데이터에 접근할 수 있다.
- 즉, random access가 불가능하다.
- tensorflow는 모든 데이터셋이 이 iterable-style 이다.
- 데이터의 양이 디스크에 올리기에는 너무 크거나, 네트워크로부터만 데이터를 넘겨받을 수 있는 경우에 사용한다.
- 데이터를 Shuffle 하거나, multi-processing 하거나, batch-size를 유동적으로 변경하는 것에 애로사항이 생긴다.


## customizing data loading order

- iterable-style 은 사용자 정의 iterable로 전부 처리해야 하기 때문에 논외.
### `torch.utils.data.Sampler`
- **map-style dataset에 대해서.**
- `DataLoader`의 `shuffle` arg에 의해서 순차적으로 데이터를 가져오는 sampler를 생성할지, 랜덤하게 가져오는 sampler를 생성할지가 결정된다.
- batch 형태의 인덱스들을 생성하고 싶으면 `batch_sampler` arg를 사용하자.
  
## automatic batching

### `collate_fn`

- 생성할 `batch`에 대해 적용하고 싶은 것이 있을 때 사용한다.
  
> [Custom collate_fn 이 필요한 이유](https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3)
   >> 위 링크의 내용을 요약하자면, data의 길이가 매번 다른 dataset의 경우, batch를 형성할 때 차원이 어그러져서 tensor가 안생기는 에러가 생긴다는 것이다.    
   >>
   >> 이를 해결하기 위해 `from torch.nn.utils.rnn` 패키지의 `pad_sequence` 메서드가 필요하다.  
   >>
   >> 해당 메서드를 이용해 가장 긴 길이의 data에 맞추어 모자란 길이들을 모두 0 으로 패딩해준다.  
   >>
   >> 전체 dataset에 대해 이걸 해버리면 패딩을 너무 많이 해줘야 해서 메모리 낭비가 발생하니, 매번 batch에 대해서만 해당 batch에서 가장 긴 data만 기준으로 할 필요가 있다. 이를 위해 **Custom collate_fn이 필요하다.**

- default collate_fn은 단순히 torch.stack 메서드를 사용한다는 듯..

- sampler -> collate_fn 의 프로세스는 단순히 다음과 같이 이해된다.

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```
- 혹은 automatic batching(단순한 이어붙이기)이 꺼졌을 때 (e.g. batch_size, batch_sampler 둘 다 None 일 때)는 이렇게!
```python
for index in sampler:
    yield collate_fn(dataset[index])
```

## single / multi process data loading

- 하드웨어적인 이야기를 하게되는데,
- 원래 python은 [GIL (Global Interpreter Lock) ](https://wiki.python.org/moin/GlobalInterpreterLock) 에 의해서 멀티 스레드를 지원하지 않는단다.
- Pytorch 는 데이터를 불러오는 과정에서 컴퓨팅이 방해받지 않도록,
- `num_workers`를 양수로 지정하는 순간 multi process data loading을 지원한다.
- single process data loading 도 나쁜건 아니다.
  - resource가 제한되어있다거나,
  - 데이터의 크기가 작아서 한번에 올릴 수 있는 양이거나,
  - 에러가 한번에 읽혀서 디버깅이 쉽다거나,

### multi process data loading
- DataLoader의 iterator가 생성될 때마다 `num_workers`만큼의 workers가 같이 생성된다.
- 각 workers에게 1. `dataset`, 2. `collate_fn`, 3. `worker_init_fn` 이 전달된다.
  - 이 말은 즉, 내부적으로 입출력과 transform들이 worker 단계에서 실행된다는 것을  의미한다.
- `torch.utils.data.get_worker_info()` 에 각 worker 들의 정보를 확인해 볼 수 있다. (Id라던가, seed라던가)
- 쓸일이 있을까 싶긴 한데,,, iterable-dataset의 경우 각 worker 들에게 `dataset`이 복사되어서 전달된다.
  - 그래서 순진하게 multi-process 를 하게 되면 데이터가 중복되는 일이 벌어진다. 
  - 각 worker들을 독립적으로 관리, 접근할 수 있는 방법이 [공식문서](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)에 잘 정리되어 있다고 한다. 
- Windows 와 프로젝트를 공유한다면, `DataLoader`의 arg들이 pickle 직렬화되어 worker들에게 넘겨지기 때문에, 다음의 사항들을 잘 지켜야 한다.
  1. `if __name__ == '__main__':` 블록으로 메인함수를 잘 감싸서, 매 worker 들이 수행될 때 메인함수가 다시 시행되는 일이 없도록 (메인함수가 확실히 한번만 실행되도록)하라. dataset, dataloader 인스턴스 생성도 이 안에서 하는게 바람직하다
  2. `custom collate_fn`, `worker_init_fn` or `custom dataset` 등은 `__main__` 밖의 최상위 경로에 선언해서 worker들이 잘 참조할 수 있게 하라.

## automatic memory pinning
- CPU에서 GPU로 데이터를 옮기는 과정을 빠르게 하기 위한 기법 정도로만 이해하고 넘어가자.