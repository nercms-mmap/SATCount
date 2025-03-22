
# SATCount
This repo is the official implementation of "SATCount: A scale-aware transformer-based class-agnostic counting framework" by Yutian Wang,Bin Yang,Xi Wang,Chao Liang,Jun Chen.

## Abstract
This paper studies the class-agnostic counting problem, which aims to count objects regardless of their class, and relies only on a limited number of exemplar objects. Existing methods usually extract visual features from query and exemplar images, compute similarity between them using convolution operations, and finally use this information to estimate object counts. However, these approaches often overlook the scale information of the exemplar objects, leading to lower counting accuracy for objects with multi-scale characteristics. Additionally, convolution operations are local linear matching processes that may result in a loss of semantic information, which can limit the performance of the counting algorithm. To address these issues, we devise a new scale-aware transformer-based feature fusion module that integrates visual and scale information of exemplar objects and models similarity between samples and queries using cross-attention. Finally, we propose an object counting algorithm based on a feature extraction backbone, a feature fusion module and a density map regression head, called SATCount. Our experiments on the FSC-147 and the CARPK demonstrate that our model outperforms the state-of-the-art methods.


## Approach

![圖一](https://github.com/user-attachments/assets/898aaffa-bd9f-4b4c-92e8-6a74f1f2eabd)

Environment Configuration Reference (https://github.com/Verg-Avesta/CounTR)

## Model training

23finetune_wyt_scale.py

## Model testing

a23FSC_test_wyt_scale.py

## Model code

a23models_wyt_scaleoldmicro.py

## Other

./data --FSC147 dataset  
├── data  
│   ├── gt_density_map_adaptive_384_VarV2  
│   ├── images_384_VarV2  
│   ├── annotation_FSC147_384.json  
│   ├── ImageClasses_FSC147.txt  
│   └── Train_Test_Val_FSC_147.json  

./output_fim6_dir --Where the trained model is stored  
