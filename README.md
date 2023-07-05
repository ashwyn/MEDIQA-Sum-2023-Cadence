Team Cadence's submission to MEDIQA-Sum-2023 shared task.

# Citation
```
@Inproceedings{MEDIQA-Sum2023,

author = {Ashwyn Sharma and David I. Feldman},

title = {Team Cadence at MEDIQA-Sum 2023: Using ChatGPT as a data augmentation tool for classifying clinical dialogue},

booktitle = {CLEF 2023 Working Notes},

series = {{CEUR} Workshop Proceedings},

year = {2023},

publisher = {CEUR-WS.org},

month = {September 18-21},

address = {Thessaloniki, Greece}

}
```

# How to setup this repo
Team Cadence's submission to MEDIQA-Sum-2023 shared task.

# How to setup this repo
```
# Setup Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs


git lfs install 

git clone git@github.com:ashwyn/MEDIQA-Sum-2023-Cadence.git

# Without lfs pull, models will not be downloaded 
git lfs pull
```

# How to run this code
```
cd code
source ./install.sh
source ./activate.sh
bash ./decode_task{A,B,C}_run{1,2,3}.sh [input-csv-file]

bash ./decode_taskA_run1.sh ../data/2023_ImageCLEFmed_Mediqa-main/dataset/TaskA/taskA_testset4participants_headers_inputConversations.csv


Output will be available in code/outputs/taskA_Cadence_run1_mediqaSum.csv


```


# Methods

- Classification: 

Augmented the dataset by generating 2722 classification examples using OpenAI's gpt-3.5-turbo model. And then trained with the augmented dataset + taskA training dataset,  facebook/bart-large for sequence classification. 