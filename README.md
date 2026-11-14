# HSCD course project: a compact student for CHAL

This repository is a course project built on top of the official CHAL code by Duan et al. (CVPR 2026,
[original repository](https://github.com/UESTC-nnLab/CHAL)). Their commits are kept at the start of the history,
and everything we changed or added comes after them. The original README follows below our section.

**What we did, on NUDT-MIRSDT only:**

- Retrained CHAL. The public code imports `nets/module/CBAM.py`, which was never published, so we wrote our own
  from the authors' attention classes. We also fixed a few loading and evaluation bugs.
- Built a compact student (1.41 M parameters, 1.99 GMACs per new frame) that runs the backbone once per frame
  and uses the median of a 4-frame buffer as its background.
- Tried feature distillation, frequency-balanced feature distillation, and a CHAL teacher with FDAM-style
  attention, all on one frozen split.

**Main results (test set, frozen split, one run each; teachers 4 epochs, students 8 epochs):**

| Model | F1 (%) | Pd (%) | Pd low contrast (%) | False alarms / frame | mAP50 (%) |
| --- | --- | --- | --- | --- | --- |
| T0: original CHAL teacher | 68.2 | 51.7 | 11.5 | 0.000 | 64.8 |
| T1: CHAL teacher with FDAM | 68.8 | 56.2 | 15.7 | 0.062 | 60.6 |
| S0: student, labels only | 90.1 | 82.9 | 59.0 | 0.010 | 75.0 |
| S1: + feature distillation | 88.5 | 80.6 | 57.5 | 0.015 | 73.2 |
| S2: + frequency-balanced distillation | 90.3 | 85.9 | 67.1 | 0.038 | 73.1 |
| S3: + frequency-balanced distillation from T1 | 87.9 | 79.2 | 50.1 | 0.009 | 74.6 |

The student runs in 17 ms per frame on a laptop CPU (AMD Ryzen AI 7 350, 8 threads), against 685 ms for the
teacher even when the teacher reuses its past backbone features.

**Limitations, plainly:** our CHAL teacher is well below the paper (our 50-epoch teacher: COCO F1 76.4 against
87.4), so these results say distillation from *our* teacher did not help, not that distillation from CHAL cannot
help. Each model was trained once and briefly, so small differences are not reliable. We also found that CHAL's
`train.py` never puts 920 of the model's own parameters into the optimizer (the Swin relative position bias,
positional embedding, frame weights and a time scale); we left this as it is for the teacher so the comparisons
stay matched. The full write-up is in `report/main.tex`.

**What is where:**

| Path | Contents |
| --- | --- |
| `nets/module/CBAM.py` | Our CBAM, built from the authors' `ChannelAttention` and `SpatialAttention` |
| `nets/student.py` | The compact student and a frame-by-frame streaming wrapper |
| `nets/distill.py` | Teacher hooks, feature and frequency-balanced distillation losses |
| `nets/stst.py` | FDAM attention inversion and frequency scaling, switched on with `fdam=True` |
| `train_student.py` | Student training (`CHAL_STUDENT_MODE` = labels, feature, frequency, distill or robust) |
| `utils_coco/nudt_to_coco.py` | NUDT-MIRSDT masks to boxes, the split, and a split manifest |
| `utils/det_metrics.py`, `utils/det_report.py` | Pd / false alarm metrics, per-sequence and low-contrast breakdown |
| `utils/background_change.py` | The label-preserving background-change test |
| `kaggle_full.py`, `kaggle_full.ipynb` | Runs the frozen-split experiments on Kaggle in two stages (teachers, then students) |
| `kaggle_train.py`, `kaggle_student.py` | Earlier Kaggle runners for the teacher and the pilot |
| `profile_models.py` | Parameters, MACs and CPU latency |
| `models/` | The four trained students from the frozen split (load with `CompactStudent(1)`) |
| `results/` | Scores, per-sequence reports, configs and timings for every run |
| `report/main.tex` | Project report (LaTeX) |

**Reproducing:** upload this code and the NUDT-MIRSDT dataset to Kaggle, open `kaggle_full.ipynb` with GPU T4 x2,
and run `STAGE = 'teachers'` (about 5-10 GPU-hours), then `STAGE = 'students'` with the first stage's output
attached (about 7-8 GPU-hours). Each stage picks its number of epochs from the measured speed.

---

# CHAL
**[CVPR 26]CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection**

![frame](readme/CHAL.png)

## Introduction
Infrared small target detection is one highly special category of object detection, faced with tiny target imaging size and cluttered backgrounds. Currently, almost all existing methods are target-centered, directly learning the target features from backgrounds. However, due to weak target signals, they are often difficult in effectively capturing stable features. Sometimes, they cannot even distinguish real targets from background confounders. To overcome these problems, from an opposite perspective, we propose the first Causal-guided Hierarchical Anomaly-aware Learning (CHAL) framework. Breaking through target-centered paradigm, it focuses on background learning, while the targets are handled as the anomalies in backgrounds. In detail, to fulfill the goal, a spatio-temporal neural field is designed to model the background evolution patterns from generative perspective. Meanwhile, a hierarchical anomaly-aware learning is proposed to decompose anomaly discovery. Furthermore, to block the spurious correlations often caused by background confounders, and enhance true target causality, a causal-guiding mechanism is designed. The experiments on three infrared datasets verify the effectiveness and superiority of our CHAL. Even in visible-light scenarios, it still possesses obvious adaptivity.

## Prerequisite
- scipy==1.10.1
- numpy==1.24.4
- matplotlib==3.7.5
- opencv_python==4.9.0.80
- torch==2.0.0+cu118
- torchvision==0.12.0
- pycocotools==2.0.7
- timm==0.9.16
- python==3.8.19
- Tested on Ubuntu 22.04.6, with CUDA 12.0, and 1x NVIDIA 4090(24 GB)

## Datasets (bounding box-based)
- Datasets are available at [DAUB-H](https://pan.baidu.com/s/18hq2ArrOSNoFh0Nq0u-ZlA) (code: xkps), [NUDT-MIRSDT](https://pan.baidu.com/s/1dYe8kKXs-yKILYo8U_Ipag?pwd=5sdr) and [IRDST-R](https://pan.baidu.com/s/1XRiP6nhWgzy8Cn0i-EssVA) (code: kt4d). DAUB-H is the hard version of DAUB, IRDST-R is the reconstruced version of IRDST.
- You need to reorganize these datasets in a format similar to the `coco_train_multi.txt` and `coco_val_multi.txt` files we provided (`.txt files` are used in training).  We provide the `.txt files` for NUDT-MIRSDT, DAUB-H and IRDST-R.
  For example:

```python
train_annotation_path = 'datasets/coco_train_multi.txt'
val_annotation_path = 'datasets/coco_val_multi.txt'
```
- The folder structure should look like this:

```
- coco_train_multi.txt
- coco_val_multi.txt
DAUB-H
├─instances_train2017.json
├─instances_test2017.json
├─coco_train_DAUBH.txt
├─coco_val_DAUBH.txt
├─images
│   ├─1
│   │   ├─0.bmp
│   │   ├─1.bmp
│   │   ├─2.bmp
│   │   ├─ ...
│   ├─2
│   │   ├─0.bmp
│   │   ├─1.bmp
│   │   ├─2.bmp
│   │   ├─ ...
│   ├─3
│   │   ├─ ...
```

## Usage of CHAL
### Train
- Note: Please change settings for different datasets, including the txt files, and image paths. Besides, you should load the pretrained model weights for further training.
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
### Test

- Usually `best_model.pth` is not necessarily the best model. Please try different models for verification.

```python
"model_path": './logs/model.pth'
```

- You need to change the path of the `json file` of test sets. For example:

```python
cocoGt_path         = 'datasets/DAUB-H/instances_test2017.json'
dataset_img_path    = 'datasets/DAUB-H/'
```

```python
python test.py
```

### Visualization

- We support `video` and `single-frame image` prediction.

```python
mode = "predict"
```

```python
python predict.py
```

## Results
- For bounding box detection, we use COCO's evaluation metrics, mAP50, Precision, Recall and F1 score.

![pr](readme/result.png)

## Contact
IF any questions, please contact with Weiwei Duan via email: [dwwuestc@163.com]().


## citation
```
@InProceedings{Duan_2026_CVPR,
    author    = {Duan, Weiwei and Ji, Luping and Lei, Shipeng and Zhu, Sicheng and Huang, Jianghong and Ye, Mao},
    title     = {CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {21357-21366}
}
```
