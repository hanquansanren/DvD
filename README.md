## DvD: Unleashing a Generative Paradigm for Document Dewarping via Coordinates-based Diffusion Model 

We present DvD, the first diffusion model for document dewarping. Unlike the existing paradigms, DVD can yield a precise yet faithful document through a novel mapping generation paradigm, where we operate coordinate-level denoising to generate coordinate mappings. We further introduce a large-scale and fine-grained benchmark, AnyPhotoDoc6300, enabling more comprehensive evaluation.



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
