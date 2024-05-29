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
