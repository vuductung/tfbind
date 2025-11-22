import numpy as np


def dna_to_one_hot_encode(dna):
    base_to_one_hot = {
        "A": (1, 0, 0, 0),
        "C": (0, 1, 0, 0),
        "T": (0, 0, 1, 0),
        "G": (0, 0, 0, 1),
        "N": (1, 1, 1, 1),  # unknown
    }
    return np.array([base_to_one_hot[base] for base in dna])
