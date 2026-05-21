import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from presidio_analizer import AnalyserEngine
from sklearn.metrics.pairwise import cosine_similarity
#CONNECTING TO THE MODEL OF TEST DATA
print("Uploading the small size model data (Pythia-70m)")
model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval()
# SIMULATION OF A REAL-WORLD SFT dataset (Input/Output)
# One of the examples intentionnally containing a Personnally Identifiable Information (PII)
data = {
  "inscription": [
    "Provide the account status for John Doe.",
    "What is the capital of France?",
    "Generate a template response for a standard client."
  ], 
  "response": [
    "The account for John Doe: (ID: 4509-AZ) is active.",
    "The capital of France is Paris",
    "Hello, your request has been successfully received."
  ],
  "user_id": [101, 102, 101], # User profiles for K-Anonymity
  "role": ["Admin", "User", "Admin"]
}
df = pd.dataframe(data)
