# PPCL
Download SMPD train dataset from http://smp-challenge.com/2023/download.html, the file structure is as follows:
```
├── dataset
│   ├── SMP_train_images
│   │   ├── 1@N18
│   │   │   ├── 1075.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── train_allmetadata_json
│   │   ├── train_additional_information.json
│   │   ├── train_category.json
│   │   ├── train_img_path.txt
│   │   ├── train_label.txt
│   │   ├── train_temporalspatial_information.json
│   │   ├── train_text.json
│   │   ├── train_user_data.json
|   ├── user_additional.txt
```
Then 

1. run ```python preprocess_data.py``` to generate preprocessed dataset
2. run ```python extract_feature.py``` to generate clip features
3. run ```python main.py``` to train the model

If you find this repo useful, welcome to cite the following paper:
```
@misc{zhang2024contrastive,
    title={Contrastive Learning for Implicit Social Factors in Social Media Popularity Prediction},
    author={Zhizhen Zhang and Ruihong Qiu and Xiaohui Xie},
    year={2024},
    eprint={2410.09345},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
}
```
