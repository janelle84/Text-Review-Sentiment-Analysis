#!/usr/bin/python

import math
import re
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

# Feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    feature_vector = defaultdict(int)
    for word in re.findall(r"\b\w+\b", re.sub(r"'s\b", "", x.lower())):
        feature_vector[word] += 1
    return dict(feature_vector)

# Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.
    """
    def predictor(x: str):
        return 1 if dotProduct(extractWordFeatures(x), weights) >= 0 else -1

    weights = {}  # the weight vector
    for epoch in range(numEpochs):
        print(f'Training Error: ({epoch} epoch): {evaluatePredictor(trainExamples, predictor)}')
        print(f'Validation Error: ({epoch} epoch): {evaluatePredictor(validationExamples, predictor)}')
        for x, y in trainExamples:
            y = 0 if y == -1 else y
            feature_vector = featureExtractor(x)
            k = dotProduct(weights, feature_vector)
            h = 1/(1+math.exp(-k))  # Sigmoid
            increment(weights, -alpha * (h - y), feature_vector)  # Gradient Descent

    return weights

# Generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrence.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        phi = {}
        for _ in range(random.randint(1, len(weights))):
            phi[random.choice(list(weights.keys()))] = 1
        y = 1 if dotProduct(phi, weights) > 0 else 0

        return phi, y

    return [generateExample() for _ in range(numExamples)]

# Extract character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:

        feature_vector = defaultdict(int)
        long_string = re.sub(r'[^a-z]', '', x.lower())
        for i in range(len(long_string) - n + 1):
            feature_vector[long_string[i:i+n]] += 1

        return feature_vector

    return extract

# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

