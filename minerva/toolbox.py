"""
Copyright 2017 Libre AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from collections import Counter, defaultdict


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


def load_wef_ground_truth(fname):
    links = defaultdict(Counter)
    is_header = True
    with open(fname, "r") as fin:
        for l in fin:
            if is_header:
                is_header = False
                continue
            source, target, type_, strength = l.strip().split("\t")
            if type_ != "Risk-Risk":
                continue
            source = source.replace("R_", "")
            target = target.replace("R_", "")
            links[source][target] += int(strength)
            links[target][source] += int(strength)
    return links


def precision(relevant, retrieved):
    return 0.0 if len(retrieved) == 0 else len(relevant.intersection(retrieved)) / len(retrieved)


def recall(relevant, retrieved):
    return 0.0 if len(relevant)  == 0 else len(relevant.intersection(retrieved)) / len(relevant)


def jaccard(relevant, retrieved):
    return len(relevant.intersection(retrieved)) / len(relevant.union(retrieved))


def compute_prec_rec_f1_jac_metrics(gt_links, topn_links, topn=1):
    prec = []
    rec = []
    jac = []
    for r in topn_links.keys():
        risks_true = set([x for x in gt_links[r].keys()])
        risks_pred = set([x[0] for x in topn_links[r]][:topn])
        prec.append(precision(risks_true, risks_pred))
        rec.append(recall(risks_true, risks_pred))
        jac.append(jaccard(risks_true, risks_pred))

    mean_prec, mean_rec, mean_jac = np.mean(prec), np.mean(rec), np.mean(jac)
    f1 = 2 * mean_prec * mean_rec / (mean_prec + mean_rec)
    return {'precision': mean_prec, 'recall': mean_rec, 'f1': f1, 'jaccard': mean_jac}
