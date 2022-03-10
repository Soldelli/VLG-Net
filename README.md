# VLG-Net: Video-Language Graph Matching Networks for Video Grounding

## Introduction
Official repository for VLG-Net: Video-Language Graph Matching Networks for Video Grounding. [[ICCVW Paper](https://openaccess.thecvf.com/content/ICCV2021W/CVEU/papers/Soldan_VLG-Net_Video-Language_Graph_Matching_Network_for_Video_Grounding_ICCVW_2021_paper.pdf)]

The paper is accepted to the first edition fo the ICCV workshop: AI for Creative Video Editing and Understanding (CVEU).

![architecture](https://user-images.githubusercontent.com/26504816/145061131-126f9341-432a-4351-ab56-679150f5f7c9.png)


## Installation

Clone the repository and move to folder:
```bash
git clone https://github.com/Soldelli/VLG-Net.git
cd VLG-Net
```

Install environmnet:
```bash
conda env create -f environment.yml
```

If installation fails, please follow the instructions in file `doc/environment.md` [(link)](doc/environment.md).

## Data
Download the following resources and extract the content in the appropriate destination folder. See table. 

| **Resource** | Download Link  | File Size | Destination Folder |
| ----         |:-----:         |:-----:    |  :-----:    |
| **StandfordCoreNLP-4.0.0** |  [link](https://drive.google.com/file/d/1lwNSgL4Xvcx-ssEBt1hNhUpcK5Zc2YRU/view?usp=sharing) | (~0.5GB) | `./datasets/`|
| **TACoS**                  |  [link](https://drive.google.com/file/d/1p7Fim1zIojGPH3gUjeQsU3YzZKTIK7Ls/view?usp=sharing) | (~0.5GB) | `./datasets/`|
| **ActivityNet-Captions**   |  [link](https://drive.google.com/file/d/11LmWxRHCOW3fhCi9usTZT2otR4hQHmrF/view?usp=sharing) | (~29GB)  | `./datasets/`|
| **DiDeMo**                 |  [link](https://drive.google.com/file/d/1-GIVOvr-zEKNe7yezYAxg37OJLdE7g4I/view?usp=sharing) | (~13GB)  | `./datasets/`|
| **GCNeXt warmup**          |  [link](https://drive.google.com/file/d/1KLuKR_Wv1-wrAL1qyzN85XN4-GTCavHV/view?usp=sharing) | (~0.1GB) | `./datasets/`|
| **Pretrained Models**      |  [link](https://drive.google.com/file/d/1r6rQHvfBNaVUQB6DPJ3hZ1dx3wN4T4Y5/view?usp=sharing) | (~0.1GB) | `./models/`  |

</br>

The folder structure should be as follows:
```
.
├── configs
│
├── datasets
│   ├── activitynet1.3
│   │    ├── annotations
│   │    └── features
│   ├── didemo
│   │    ├── annotations
│   │    └── features
│   ├── tacos
│   │    ├── annotations
│   │    └── features
│   ├── gcnext_warmup
│   └── standford-corenlp-4.0.0
│
├── doc
│
├── lib
│   ├── config
│   ├── data
│   ├── engine
│   ├── modeling
│   ├── structures
│   └── utils
│
├── models
│   ├── activitynet
│   └── tacos
│
├── outputs
│
└── scripts
```

## Training

Copy paste the following commands in the terminal. </br>

Load environment: 
```bash
conda activate vlg
```

- For ActivityNet-Captions dataset, run:
```bash
python train_net.py --config-file configs/activitynet.yml OUTPUT_DIR outputs/activitynet
```

- For TACoS dataset, run: 
```bash
python train_net.py --config-file configs/tacos.yml OUTPUT_DIR outputs/tacos
```

## Evaluation
For simplicity we provide scripts to automatically run the inference on pretrained models. See script details if you want to run inference on a different model. </br>

Load environment: 
```bash
conda activate vlg
```

Then run one of the following scripts to launch the evaluation. 

- For ActivityNet-Captions dataset, run:
```bash
    bash scripts/activitynet.sh
```

- For TACoS dataset, run: 
```bash
    bash scripts/tacos.sh
```

## Expected results:
After cleaning the code and fixing a couple of minor bugs, performance changed (slightly) with respect to reported numbers in the paper. See below table.

| **ActivityNet** | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-----:|:-----:|:-----:|:-----:|
| **Paper**   |  **46.32** | **29.82** |  **77.15** | **63.33** |
| **Current** |  **46.32** | **29.79** |  **77.19** | **63.36** |
</br>

| **TACoS** | Rank1@0.1 | Rank1@0.3 | Rank1@0.5 | Rank5@0.1 | Rank5@0.3 | Rank5@0.5 |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| **Paper**   | **57.21** | **45.46** | **34.19** | **81.80** | **70.38** | **56.56** |
| **Current** | **57.16** | **45.56** | **34.14** | **81.48** | **70.13** | **56.34** |
</br>


## Citation
If any part of our paper and code is helpful to your work, please  cite with:
```
@inproceedings{soldan2021vlg,
  title={VLG-Net: Video-language graph matching network for video grounding},
  author={Soldan, Mattia and Xu, Mengmeng and Qu, Sisi and Tegner, Jesper and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3224--3234},
  year={2021}
}
```
