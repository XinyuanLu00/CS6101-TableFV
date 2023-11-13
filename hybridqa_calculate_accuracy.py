import json
import re
import nltk
from rouge import Rouge

# nltk.download('punkt')  # Uncomment this line if you haven't downloaded the 'punkt' tokenizer models

# Function to read a JSON file
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to extract the answer from the model's output
def extract_model_answer(output, regex_pattern):
    match = re.search(regex_pattern, output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# Read a text file and return its content as a list of JSON objects
def read_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return [json.loads(line) for line in lines]

path_to_model_output = '/home/luxinyuan/CS6101-TableFV/hybridqa_test.out'
path_to_ground_truth = '/home/luxinyuan/CS6101-TableFV/processed_seed_datasets/hybridqa/test_new.json'


# Load model outputs
model_outputs = read_json_lines(path_to_model_output)

# Load ground truths
ground_truths = read_json_file(path_to_ground_truth)

# Convert ground truths to a dictionary for easy access
ground_truth_dict = {item['id']: item['answer'] for item in ground_truths}

# Initialize Rouge scorer
rouge = Rouge()

# Initialize lists for BLEU and ROUGE scores
bleu_scores = []
rouge_scores = []

# Define the regex pattern to extract model answers
regex_pattern = r"the answer is \s*(.*)"

# Compare model output with ground truth
for output in model_outputs:
    model_answer = extract_model_answer(output['output'][0], regex_pattern)
    ground_truth_answer = ground_truth_dict.get(output['id'])
    
    if model_answer is not None and ground_truth_answer is not None:
        # Tokenize sentences for BLEU
        reference = nltk.word_tokenize(ground_truth_answer.lower())
        candidate = nltk.word_tokenize(model_answer.lower())

        # Calculate BLEU score
        bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu_score)

        # Calculate ROUGE score
        rouge_score = rouge.get_scores(model_answer, ground_truth_answer)
        rouge_scores.append(rouge_score)

# Calculate average BLEU and ROUGE scores
average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
average_rouge = {
    'rouge-1': sum([score[0]['rouge-1']['f'] for score in rouge_scores]) / len(rouge_scores),
    'rouge-2': sum([score[0]['rouge-2']['f'] for score in rouge_scores]) / len(rouge_scores),
    'rouge-l': sum([score[0]['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores)
} if rouge_scores else {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

print(f"Average BLEU Score: {average_bleu:.4f}")
print("Average ROUGE Scores:")
for key, value in average_rouge.items():
    print(f"  {key.upper()}: {value:.4f}")
