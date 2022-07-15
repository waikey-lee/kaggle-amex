# ============================================================================
# Import Package
# ============================================================================
import numpy as np

# ============================================================================
# Define Functions
# ============================================================================ 
def sigmoid(log_odds):
    return 1 / (1 + np.exp(-log_odds))

def reverse_sigmoid(prob):
    return -np.log((1 / prob) - 1)

def pad_column_name(columns, suffix, prefix=""):
    return [prefix + column + suffix for column in columns]