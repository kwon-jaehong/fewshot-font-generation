LFfont는 페이즈1 페이즈2 학습을 진행함 ( 학습 스테이지가 2개임 )

```
python train_LF.py cfgs/LF/p1/train.yaml cfgs/data/train/custom.yaml --phase 1 --work_dir ./temp/outputs

python train_LF.py cfgs/LF/p1/train.yaml kor_train/kor_train.yaml --phase 1 --work_dir ./temp/outputs

```

학습할때 cfgs/LF/p1/train.yaml는 자연스럽게 default 참조함


```
python train_LF.py cfgs/LF/p2/train.yaml cfgs/data/train/custom.yaml --resume temp/outputs/checkpoints/last.pth --phase 2 --work_dir temp/outputs2

python train_LF.py cfgs/LF/p2/train.yaml kor_train/kor_train.yaml --resume temp/outputs/checkpoints/last.pth --phase 2 --work_dir temp/outputs2
```