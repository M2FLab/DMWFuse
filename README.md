# DMWFuse: A Universal Framework for Degradation-Adaptive RGB-IR Image Fusion

Keming Bai, Linyuan He, Shiping Ma, Jiahao Dang, Kun Liu, Mingzhao Han, and Xiaoyu Cai

- [*[Paper]*]
- [*[GitHub]*]()


---

### Datasets

- **Source**: All images are from the DeMMI-RF dataset, which is selected from the following datasets:
   - [MSRS](https://github.com/Linfeng-Tang/MSRS)
   - [M3FD](https://github.com/JinyuanLiu-CV/TarDAL)
   - [LLVIP](https://github.com/bupt-ai-cz/LLVIP)
   - [DroneVehicle](https://github.com/VisDrone/DroneVehicle)
   - [DeMMI-RF](https://github.com/LeeX54946/TG-ECNet)
---

### Download

- [Baidu Yun](https://pan.baidu.com/s/1KbaGUXzuOW6ej4maHN5ZcQ?pwd=TGEC)

---

<h2> <p align="center"> DMWFuse </p> </h2>  

---

## Set Up on Your Own Machine

---

When you want to dive deeper or apply it on a larger scale, you can configure our DMWFuse on your computer following the steps below.

We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n dmwfuse 
conda activate dmwfuse
# install dmwfuse requirements
pip install -r requirements.txt
```
---

## Quick Start

---

If you want to test the performance of fusion using this method, ensure `stage2/ckpt/stage2_pretrained.ckpt`.
Place your data in `stage2/test.py` to test your own data. You need to place the data as follows:
```
data
└── test
    ├── denoise
    |   ├── your dataname1
    |   |   ├── input # images with degradations
    |   |   ├── visible # clean visible images to provide color information
    |   |   └── infrared # infrared images
    |   └── your dataname2
    |       └──...
    ├── dehaze
    |   ├── your dataname3
    |   |   ├── input # images with degradations
    |   |   ├── visible # clean visible images to provide color information
    |   |   └── infrared # infrared images
    |   └── your dataname4
    |       └──...
    ├── deblur
    |   └── ...
    └── stripe
        └── ...
```
Then, modify `stage2/test.py` to test your own data.

```shell
conda activate dmwfuse
python stage2/test.py
```
And the result will be in `stage2/output/`.

## Train

### Data Preparation

You should put the data in the correct place in the following form.

```
data
└── Train
    ├── degrad
    |   ├── noise15
    |   |   ├── LLVIP
    |   |   ├── M3FD
    |   |   └── ...
    |   ├── noise25
    |   ├── noise50
    |   ├── haze
    |   ├── DefocusBlur
    |   └── stripe
    |       └──...
    ├── visible
    |   └── ...
    └── infrared
        └── ...

```
### Stage Ⅰ

Before training the model, you need to modify the `stage1/options.py` and the `txt` file in `stage1/data_dir`.
```shell
conda activate dmwfuse
python stage1/train.py
```
And then you should run `stage1/test.py` with the obtained `stage1/ckpt/stage1_pretrained.ckpt` to obtain  the `stage1.pth` which can be used in Stage 2.
We also offer a pretrained edition as `stage2/stage1.pth`.

---

### Stage Ⅱ

Before training the model, you need to modify the `stage2/options.py` and the `txt` file in `stage2/data_dir`.
```shell
conda activate dmwfuse
python stage2/train.py
```
And then you should run `stage2/test.py` with the obtained `stage2/ckpt/stage2_pretrained.ckpt` to obtain the outputs.

---

We offer the pretrained model parameters, you can place them like this:

```
DMWFuse
├── ckpt
|   └── stage1_pretrained.ckpt
├── stage1
└── stage2
    ├── ckpt
    |   └── stage2_pretrained.ckpt
    └── stage1.pth
```

- [Baidu Yun](链接: https://pan.baidu.com/s/1Jn7DGy72G1lorj_fOFPxtg?pwd=qxep 提取码: qxep)

---

### Any Question

If you have any other questions about the code, please email `hal1983@163.com`.


## Citation

If this work has been helpful to you, please feel free to cite our paper!


