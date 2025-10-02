## DvD: Unleashing a Generative Paradigm for Document Dewarping via Coordinates-based Diffusion Model 
<p align="center">
  <a href="https://huggingface.co/datasets/hanquansanren/AnyPhotoDoc6300">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Benchmark-90EE90?style=flat" alt="Benchmark">
  </a>
  <a href="https://huggingface.co/datasets/hanquansanren/AnyPhotoDoc6300">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Benchmark-90EE90?style=flat" alt="Benchmark">
  </a>
  <a href="https://arxiv.org/abs/2505.21975">
    <img src="https://img.shields.io/badge/DvD paper-d4333f?logo=arxiv&logoColor=white&colorA=cccccc&colorB=d4333f&style=flat" alt="Paper">
  </a>
  <a href="https://komarev.com/ghpvc/?username=hanquansanren&repo=DvD&color=brightgreen&label=Views" alt="view">
    <img src="https://komarev.com/ghpvc/?username=hanquansanren&repo=DvD&color=brightgreen&label=Views" alt="view">
  </a>
</p>

We present DvD, the first diffusion model for document dewarping. Unlike the existing paradigms, DVD can yield a precise yet faithful document through a novel mapping generation paradigm, where we operate coordinate-level denoising to generate coordinate mappings. We further introduce a large-scale and fine-grained benchmark, AnyPhotoDoc6300, enabling more comprehensive evaluation.

https://github.com/user-attachments/assets/a2c5bca3-1393-410f-b870-8e69142bc635


### Quick Start
Before running the script, install the following dependencies:

```shell
pip install -r requirements.txt
```


### How to play
To run the DVD model as shown above:

#### Download link of Pretrained DvD model 
https://drive.google.com/drive/folders/1RBt9t_5igAlN1BlQAkVLwJ_rZXITy_pN?usp=sharing


#### Inference code
```bash
python run_sampling.py \
  --train_module 'dvd' 
  --train_name 'val_TDiff' --name "save_filename"
```

#### Training code
```bash
mpiexec -n 1 python run_training.py \  
  --train_module 'dvd' 
  --train_name 'train_TDiff' 
```

### Download link of inference results in DocUNet and DIR300 benchmarks

https://drive.google.com/drive/folders/1WNMcXx4OApIy789F-iO3X2dJmmDUK_QI?usp=drive_link

### Download link of AnyPhotoDoc6300 benchmark dataset 

- HuggingFace

[https://drive.google.com/drive/folders/1NJZtaTh4erKFcXDcs1t1JyF2hc1EBOA3?usp=drive_link](https://huggingface.co/datasets/hanquansanren/AnyPhotoDoc6300)
