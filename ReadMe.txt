================================================================
CHAT REPLY RECOMMENDATION SYSTEM - DOCUMENTATION
================================================================

PROJECT OVERVIEW
---------------
Offline AI system that predicts User A's replies to User B's
messages using fine-tuned GPT-2 Transformer model.

SYSTEM REQUIREMENTS
------------------
Python: 3.10+
RAM: 8GB minimum (16GB recommended)
Storage: 2GB
GPU: Optional (3x faster training)

DEPENDENCIES
-----------
transformers
torch
pandas
numpy
scikit-learn
nltk
rouge
joblib

INSTALLATION
-----------
# Install dependencies offline
pip install transformers torch pandas numpy scikit-learn nltk rouge-score joblib

CONFIGURATION
-------------
- PATH_A: Path to User A's chat dataset (e.g., /Desktop/Dataset/userA_chats.csv)
- PATH_B: Path to User B's chat dataset (e.g., /Desktop/Dataset/userB_chats.csv)
- RUN_TRAINING: Set to True to train the model, False to skip training.
- DEV_RUN: Set to True for a quick test run with a small dataset.

EXECUTION STEPS
--------------
1. Open `ChatRec_Model.ipynb` in Jupyter Notebook.
2. Update `PATH_A` and `PATH_B` to point to your datasets.
3. Set `RUN_TRAINING` and `DEV_RUN` flags as needed.
4. Run all cells sequentially.
5. The model will save automatically as `Model.joblib`.

NOTE: Ensure `rouge-score` is installed for evaluation metrics.

DATASET FORMAT
-------------
File 1: /Desktop/Dataset/userA_chats.csv
File 2: /Desktop/Dataset/userB_chats.csv

Required Columns:
- timestamp (datetime)
- message (string)
- user_id (string: 'A' or 'B')

FILE DESCRIPTIONS
----------------
ChatRec_Model.ipynb - Main training notebook
Model.joblib - Trained model + pipeline
Report.pdf - Technical documentation
ReadMe.txt - This file

MODEL DETAILS
-------------
Base Model: GPT-2 (124M parameters)
Context Window: Last 3 messages (256 tokens)
Generation: Nucleus sampling (top-p=0.95, temp=0.8)
Training: 3 epochs, batch size 4

USAGE EXAMPLE
------------
import joblib
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
pipeline = joblib.load('Model.joblib')
model = GPT2LMHeadModel.from_pretrained('./final_model')
tokenizer = GPT2Tokenizer.from_pretrained('./final_model')

# Generate reply
context = "B: Hello | A: Hi | B: How are you?"
input_text = f"{context} <SEP> A:"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=50)
reply = tokenizer.decode(output[0], skip_special_tokens=True)
print(reply.split("A:")[-1])

PERFORMANCE
----------
Training Time: 60-90 minutes
Inference: <2 seconds per reply
Memory Usage: ~2GB during inference

EVALUATION METRICS
-----------------
BLEU - Text generation quality
ROUGE-1/2/L - Semantic similarity
Perplexity - Model confidence

TROUBLESHOOTING
--------------
Error: CUDA out of memory
Fix: Set fp16=False in TrainingArguments or use CPU

Error: Module not found
Fix: pip install [missing_module]

Error: Dataset not found
Fix: Verify paths to CSV files

CONTACT
-------
Developed for: Meetmux AI/ML Internship Assessment Round
Framework: Hugging Face Transformers
Base Model: OpenAI GPT-2

================================================================
