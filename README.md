# CS6101-TableFV-LLAMA
This is the code base for CS6101 Project: Evaluating Open-Sourced LLM for Table Fact-Checking and Reasoning

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
