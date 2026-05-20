import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausaLLM
from presidio_analizer import AnalyserEngine
from sklearn.metrics.pairwise import cosine_similarity
