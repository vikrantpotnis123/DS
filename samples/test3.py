import numpy
import math

# Set a random seed so the shuffle is the same every time
numpy.random.seed(1)

# Shuffle the rows
# This permutes the index randomly using numpy.random.permutation
# Then, it reindexes the dataframe with the result
# The net effect is to put the rows into random order
income = income.reindex(numpy.random.permutation(income.index))

train_max_row = math.floor(income.shape[0] * .8)
train = income[0:train_max_row]
test = income[train_max_row:]