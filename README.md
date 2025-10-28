Repo for Privacy Informed Mental Bert

# Directory Structure
    .
    ├── data
    │   ├── processed
    │   │   └── Dreaddit\\_dataset
    │   └── raw
    │       ├── Depression\\_Reddit\\_dataset
    │       ├── Dreaddit\_dataset
    │       ├── SAD\\_dataset
    │       ├── SWMH
    │       └── SWMH\\_dataset
    ├── json
    ├── logs
    │   └── Dreaddit\_No\_DP\_finetune
    │       └── lr=0.01
    │           └── epochs=10
    │               └── scheduler=Linear
    │                   └── optimizer=AdamW
    │                       └── AdamW\_epsilon=10**-8
    ├── logs\_clf\_dreddit
    ├── model\_classifier
    ├── model\_classifier\_dreaddit
    ├── src
    │   └── \_\_pycache\_\_
    └── yaml

## src directory:
    
    src
    ├── \_\_pycache\_\_
    │   └── utils.cpython-311.pyc
    ├── dp\_finetune\_model.py
    ├── process\_Depression\_Reddit\_dataset.py
    ├── process\_Dreaddit\_dataset.py
    ├── process\_SAD\_dataset.py
    ├── utils.py
    └── validate\_model\_performance.py


# Training: 
    * To dp finetune model: 
        1. Open json/training\_hyperparams.json
        2. Change output dir key to model\_clf\_dp\_\<dataset\\_name\>
        3. Change logging dir key to logs\_clf\_dp\_\<dataset\\_name\>
        4. In bash run export HF\\_TOKEN=\<hf\\_token\\_dir\>
        5. Run python src/dp\_finetune\_model.py --model\\_name "mental/mental-bert-base-uncased" --data\_dir "data/processed/\<model\\_name\>" --seed \<seed\>
        6. Run above for each seed (42, 0, 1, 1234, 100)

    * No dp finetune model:
        1. Open json/training\_hyperparams.json
        2. Change output dir key to model\_clf\_no\_dp\_\<dataset\\_name\>
        3. Change logging dir key to logs\_clf\_no\_dp\_\<dataset\\_name\>
        4. In bash run export HF\_TOKEN=\<hf\\_token\\_dir\>
        5. Run python src/validate\_model\_performance --model\\_name "mental/mental-bert-base-uncased" --data\_dir "data/processed/\<model\\_name\>" 

# Inference:

    python src/inference.py --data\_dir data/processed/\<dataset\\_name\>


# Overall Pipeline: 

 1. Follow Training and Inference instructions for SAD and Dreaddit datasets
 2. Check files json files for relevant info:
    * Inference Results: clf\\_report.json and test\\_results.json 
    * Training Metadata: training\\_metadta.json
  
 

