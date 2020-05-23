import numpy

def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = numpy.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)

    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            entropy += prob * numpy.math.log(prob, 2)

    return -entropy


# Verify that our function matches our answer from earlier
entropy = calc_entropy([1, 1, 0, 0, 1])
print(entropy)

information_gain = entropy - ((.8 * calc_entropy([1, 1, 0, 0])) + (.2 * calc_entropy([1])))
print(information_gain)
income_entropy = calc_entropy(income["high_income"])

median_age = income["age"].median()

left_split = income[income["age"] <= median_age]
right_split = income[income["age"] > median_age]

age_information_gain = income_entropy - (
            (left_split.shape[0] / income.shape[0]) * calc_entropy(left_split["high_income"]) + (
                (right_split.shape[0] / income.shape[0]) * calc_entropy(right_split["high_income"]))