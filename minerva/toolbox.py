import numpy as np

def sigmoid(x):
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distance(centroid, vector):
    return np.sqrt(np.sum((vector - centroid)**2))


def compute_median_and_mad(centroid, vectors):
    distances = np.sqrt(np.sum((vectors - centroid)**2, axis=1))
    median = np.median(distances)
    absolute_deviations_from_median = np.sqrt((distances - median)**2)
    mad = np.median(absolute_deviations_from_median)
    return median, mad


def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))/temperature)
    return e_x / e_x.sum()

