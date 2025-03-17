
# SATCount: A scale-aware transformer-based class-agnostic counting framework

This project is based on CounTR, and the new model is called SATCount.

Environment Configuration Reference（https://github.com/Verg-Avesta/CounTR）

## Model training

23finetune_scale.py

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
23checkpoint33-666.pth --The trained model corresponding to the optimal result
