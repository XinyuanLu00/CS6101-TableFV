def build_table_row(row):
    return f"||{'|'.join(row)}||"

def get_caption(data):
    if 'table_caption' not in data:
        return []
    if data['table_caption'] is None:
        return []
    return [
        f"Caption: {data['table_caption']}"
    ]

def format_prompt(data):
    table_header = build_table_row(data['table_column_names'])
    table_body = [build_table_row(r) for r in data['table_content_values']]
    lines = [
        'Read the Instruction below and provide an answer.',
        '### INSTRUCTION:\nRead the following table and answer a question\n',
        *get_caption(data),
        'Table:',
        table_header,
        *table_body,
        *format_context(data),
        '\nQuestion:',
        data['question'],
        '\n### Response:'
    ]

    return '\n'.join(lines)

def format_answer(data):
    f"The answer is {data['answer']}"

def filter_empty_sentences(rows):
    return [r for r in rows if r != '.']

def format_context(data):
    pre = filter_empty_sentences(data['pre_text']) if 'pre_text' in data else []
    post = filter_empty_sentences(data['post_text']) if 'post_text' in data else []
    total_len = len(pre)+len(post)
    if total_len == 0:
        return []
    lines = [ '\nContext: ']
    if len(pre) > 0:
        lines.append('.'.join(pre))
    if len(post) > 0:
        lines.append('.'.join(post))
    
    return lines

def formatting_prompts_func(data):    
    return [format_prompt(d) for d in data]

# Usage with trainer
# trainer = SFTTrainer(
#     model,
#     train_dataset=dataset,
#     formatting_func=formatting_prompts_func,
# )