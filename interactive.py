"""
File: interactive.py
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from submission import *
WEIGHT_FILE = 'weights'


def main():

	weights = {}

	with open(WEIGHT_FILE, 'r') as f:
		for line in f:
			key, value = line.strip().split('\t')
			weights[key] = float(value)

	interactivePrompt(extractWordFeatures, weights)


if __name__ == '__main__':
	main()
