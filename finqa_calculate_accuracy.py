import json
import re

# Function to read a JSON file
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
# Function to extract the answer from the model's output
def extract_model_answer(output):
    match = re.search(r"the answer is\s*([\d\.]+)", output, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# Read a text file and return its content as a list of JSON objects
def read_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return [json.loads(line) for line in lines]

path_to_model_output = '/home/luxinyuan/CS6101-TableFV/finqa_examples.out'
path_to_ground_truth = '/home/luxinyuan/CS6101-TableFV/processed_seed_datasets/finqa/finqa_processed_test.json'
# Load model outputs
model_outputs = read_json_lines(path_to_model_output)

# Load ground truths
ground_truths = read_json_file(path_to_ground_truth)

# Convert ground truths to a dictionary for easy access
ground_truth_dict = {item['id']: item['answer'] for item in ground_truths}

# Initialize variables for accuracy calculation
correct_answers = 0
total_answers = 0

# Compare model output with ground truth
for output in model_outputs:
    model_answer = extract_model_answer(output['output'][0])
    ground_truth_answer = ground_truth_dict.get(output['id'])
    
    if model_answer and ground_truth_answer:
        total_answers += 1
        if model_answer == ground_truth_answer:
            correct_answers += 1

# Calculate accuracy
accuracy = (correct_answers / total_answers) * 100 if total_answers > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
