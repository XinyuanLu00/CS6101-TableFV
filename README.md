# CS6101-TableFV
This is the code base for CS6101 Project: Evaluating Open-Sourced LLM for Table Fact-Checking and Reasoning

![alt text](https://github.com/XinyuanLu00/CS6101-TableFV/blob/main/Poster.png?raw=true)

## Dataset

Each seed dataset is stored as json files in folder "processed_seed_dataset", each entry has the following format:

```bash
"id": the sample id
"table_caption": the table caption
"table_column_names": the table column names
"table_content_values": the table content
"question": the question texts
"answer": the answer for each question
"context": the input of hybridqa, tatqa, finqa also contains context paragraph.
```

For hybridqa training set,tabfact training set,wikitablequestion training set, please download from the Google Drive [Link](https://drive.google.com/drive/folders/1IH6dep2eQvz9Lw_Iz9XnqcCuX69EhB_R?usp=sharing).

## LLAMA2-Finetune
```bash
In `llama_finetune` folder, it contains
    --lora_inference.py 
    --my_lora_trainer.py 
    --train.sh
In train.sh, please replace MYDIR with your own path
```