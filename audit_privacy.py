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
  "instruction": [
    "Provide the account status for Thierry Kamgang.",
    "What is the capital of United Kingdom?",
    "Generate a template response for a standard client."
  ], 
  "response": [
    "The account for Thierry Kamgang: (ID: 4509-AZ) is active.",
    "The capital of United Kingdom is London",
    "Hello, your request has been successfully received."
  ],
  "user_id": [101, 102, 101], # User profiles for K-Anonymity
  "role": ["Admin", "User", "Admin"]
}
df = pd.dataframe(data)
# EXPOSURE AUDIT (LEAKAGE VIA ATTENTION/INVERSION)
def audit_exposure(text_to_test, model, tokenizer):
  print(f"\n[Audit] Exposure Audit on : '{text_to_test}'")
#Context vector extraction
inputs = tokenizer(text_to_test, return tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)
    #Fetching the hidden states of the last layer
    hidden_states = outputs.hidden_states[-1]
    mean_embedding = hidden_states.mean(dim=1).numpy()
  with torch.no_grad():
    logits = outputs.logits[:, -1, :]
    top_tokens = torch.topk(logits, k=5).indices[0].tolist()
    predicted_words = [tokenizer.decode([tok] for tok in top_tokens]
#PII scanning in textual reconstruction via Microsoft Presidio
  analyzer = AnalyserEngine()
  analysis_results = analyzer.analyze(text=text_to_test, language="en")
  pii_detected = len(analysis_results) > 0
  print(f"–> Most probable words reconstructed by the model:{predicted_words}")
  print(f"–> PII EXPOSURE RISK DETECTED BY PRESIDIO : {'Yes' if pii_detected else 'No'}")
  return pii_detected, mean_embedding
# K-ANONYMITY CALCULATION
def calculate_k_anonymity(dataframe, quasi_identifiers):
  print(f"\n[Audit] k-anonymity calculation for columns: {quasi_identifiers}")
  grouped = dataframe.groupby(quasi_identifiers).size().reset_index(name='counts')
    k-value = grouped['counts'].min()
    print(grouped)
    return k_value
#FULL AUDIT EXECUTION
if __name__ == "__main__":
  sample_text = df["response"].iloc[0] #first testing the first row to monitor to check whether the full process is working before applying to...
  pii_found, _ = audit_exposure(sample_text, model, tokenizer)
  #Run K-anonymity check on Quasi-identifiers User_id and role
  k_min = calculate_k_anonymity(df, ["user_id", "role"])
  print("n\" + "="*40)
  print("FINAL PRIVACY REPORT")
  print("="*40)
  if pii_found: 
    print("AlERT Unmasked PII detected. High risk of model memorization!")
  else: 
    print("[SAFE] No direct PII detected on model outputs -- CCPA/GDPR COMPLIANCE OK")
  print(f"[STATUS] Dataset k-anonymity score: k = {k_min}")
  if k_min < 5:
    print("[DANGER] K < 5 UNIQUE PROFILE DETECTED, HIGH RE-IDENTIFICATION RISK -- CCPA/GDPR COMPLIANCE FAILED")
  else: 
    print("[SAFE] STATISTICAL ANONIMITY LEVEL IS ACCEPTABLE")



  


  

    
