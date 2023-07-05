#!/usr/bin/env python
# coding: utf-8

# In[2]:
import sys 
import os

os.system('conda run -n Cadence_tasks_venv python --version')

VAL_MODE=False


# In[3]:


val_filename = '../data/2023_ImageCLEFmed_Mediqa-main/dataset/TaskA/TaskA-ValidationSet.csv'


# In[4]:


# Pass as CLI arg
# test_filename = '../data/2023_ImageCLEFmed_Mediqa-main/dataset/TaskA/taskA_testset4participants_headers_inputConversations.csv'
test_filename = sys.argv[1]


# In[5]:


if VAL_MODE:
    test_filename = val_filename


print(f'Test input filename:{test_filename}\n')


# In[7]:


output_filename = './outputs/taskA_Cadence_run1_mediqaSum.csv'


temp_classifier_output_filename = './outputs/temp/taskA-classifier/classifier-taskA_Cadence_run1.csv'


import pandas as pd
def read_csv(filename):
    return pd.read_csv(filename)


# In[15]:


val_df = read_csv(val_filename)
val_df



import evaluate 
rouge = evaluate.load('rouge')


# Classification
idx2label = {0: 'PROCEDURES',
 1: 'OTHER_HISTORY',
 2: 'IMMUNIZATIONS',
 3: 'DIAGNOSIS',
 4: 'PASTMEDICALHX',
 5: 'PASTSURGICAL',
 6: 'LABS',
 7: 'ALLERGY',
 8: 'EDCOURSE',
 9: 'PLAN',
 10: 'FAM/SOCHX',
 11: 'GYNHX',
 12: 'DISPOSITION',
 13: 'MEDICATIONS',
 14: 'IMAGING',
 15: 'GENHX',
 16: 'ROS',
 17: 'ASSESSMENT',
 18: 'CC',
 19: 'EXAM'}

label2idx = {'PROCEDURES': 0,
 'OTHER_HISTORY': 1,
 'IMMUNIZATIONS': 2,
 'DIAGNOSIS': 3,
 'PASTMEDICALHX': 4,
 'PASTSURGICAL': 5,
 'LABS': 6,
 'ALLERGY': 7,
 'EDCOURSE': 8,
 'PLAN': 9,
 'FAM/SOCHX': 10,
 'GYNHX': 11,
 'DISPOSITION': 12,
 'MEDICATIONS': 13,
 'IMAGING': 14,
 'GENHX': 15,
 'ROS': 16,
 'ASSESSMENT': 17,
 'CC': 18,
 'EXAM': 19}


# In[20]:


test_df = read_csv(test_filename)
test_df


# In[21]:


classifier_model_path = '../taskA-class-gpt3_5_augmented'

from transformers import pipeline
classifier = pipeline('text-classification', model=classifier_model_path, max_length=1024, truncation=True, padding=True)


test_dialogues = list(test_df['dialogue'])

classes_preds = classifier(test_dialogues)

classes_preds = [pred_result['label'] for pred_result in classes_preds]

classes_idx_preds = [label2idx[lab] for lab in classes_preds]


if VAL_MODE:
    classes_labels = list(test_df['section_header'])
    print(f'Total class labels:{len(classes_labels)}')
    
    classes_idx_labels = [label2idx[lab] for lab in classes_labels]

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    acc = accuracy.compute(predictions=classes_idx_preds, references=classes_idx_labels)
    f1_score = f1.compute(predictions=classes_idx_preds, references=classes_idx_labels, average='weighted')['f1']
    f1_score_non_wt = f1.compute(predictions=classes_idx_preds, references=classes_idx_labels, average=None)['f1']
    f1_score_macro = f1.compute(predictions=classes_idx_preds, references=classes_idx_labels, average='macro')['f1']

    print(f'Accuracy: {acc} | F1: {f1_score} | F1 (non-weighted): {f1_score_non_wt} | F1 (macro): {f1_score_macro}')


# In[22]:


id_column = 'TestID' if 'TestID' in test_df.columns else 'ID'


# In[23]:


taskA_out_df = test_df[[id_column]]
taskA_out_df['SystemOutput'] = classes_preds
taskA_out_df.rename(columns={id_column:'TestID'}, inplace=True)
taskA_out_df


# In[24]:


taskA_out_df.to_csv(output_filename, index=False)


# In[ ]:


print('Task A inference completed!')
