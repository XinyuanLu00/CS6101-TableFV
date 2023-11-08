def build_table_row(row):
    return f"||{'|'.join(row)}||"


def get_field(data, field, index=None):
    res = data[field]
    if index is not None:
        res = res[index]
    return res

def get_caption(data, index=None):
    if 'table_caption' not in data:
        return []
    caption = get_field(data, 'table_caption', index)
    if caption is None:
        return []
    return [
        f"Caption: {caption}"
    ]

def format_prompt(data, index=None):
    table_header = build_table_row(get_field(data, 'table_column_names', index))
    table_body = [build_table_row(r) for r in get_field(data, 'table_content_values', index)]
    lines = [
        'Read the Instruction below and provide an answer.',
        '### INSTRUCTION:\nRead the following table and answer a question\n',
        *get_caption(data, index),
        'Table:',
        table_header,
        *table_body,
        *format_context(data, index),
        '\nQuestion:',
        get_field(data, 'question', index),
        '\n### Response:'
    ]

    return '\n'.join(lines)

def format_prompt_with_answer(data, index=None):
    prompt = format_prompt(data, index)
    answer = format_answer(data, index)
    return f"{prompt}\n{answer}"

def format_answer(data, index=None):
    return f"The answer is {get_field(data, 'answer', index)}"

def filter_empty_sentences(rows):
    return [r for r in rows if r != '.']

def format_context(data, index=None):
    context = get_field(data, 'context', index) if 'context' in data else None
    pre = filter_empty_sentences(get_field(data, 'pre_text', index)) if 'pre_text' in data else []
    post = filter_empty_sentences(get_field(data, 'post_text', index)) if 'post_text' in data else []
    total_len = len(pre)+len(post)
    if total_len == 0 and context is None:
        return []
    lines = [ '\nContext:']
    if context is not None:
        lines.append(context)
    if len(pre) > 0:
        lines.append('.'.join(pre))
    if len(post) > 0:
        lines.append('.'.join(post))
    
    return lines

def formatting_prompts_func(data):
    out = []
    for i in range(len(data['question'])):
        out.append(format_prompt_with_answer(data, i))
    return out

# Usage with trainer
# trainer = SFTTrainer(
#     model,
#     train_dataset=dataset,
#     formatting_func=formatting_prompts_func,
# )