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
            entropy += prob * math.log(prob, 2)

    return -entropy


# Verify that our function matches our answer from earlier
entropy = calc_entropy([1, 1, 0, 0, 1])
print(entropy)

information_gain = entropy - ((.8 * calc_entropy([1, 1, 0, 0])) + (.2 * calc_entropy([1])))
print(information_gain)

# print(income["age"])
m = numpy.median(income["age"])
# print(m)
l = income[income["age"] <= m]
r = income[income["age"] > m]
# print(l["high_income"])
# print()
# print(r["high_income"])
income_entropy = calc_entropy(income["high_income"])
# print(income_entropy)
l_wt = l["high_income"].shape[0] / income["high_income"].shape[0]
r_wt = r["high_income"].shape[0] / income["high_income"].shape[0]
# print(l_wt)
# print(r_wt)
age_information_gain = income_entropy - (
            (l_wt * calc_entropy(l["high_income"])) + (r_wt * calc_entropy(r["high_income"])))
print(age_information_gain)


def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a data set, column to split on, and target
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])

    # Find the median of the column we're splitting
    column = data[split_name]
    median = column.median()

    # Make two subsets of the data, based on the median
    left_split = data[column <= median]
    right_split = data[column > median]

    # Loop through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0])
        to_subtract += prob * calc_entropy(subset[target_name])

    # Return information gain
    return original_entropy - to_subtract


# Verify that our answer is the same as on the last screen
print(calc_information_gain(income, "age", "high_income"))

columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex",
           "hours_per_week", "native_country"]
information_gains = [calc_information_gain(income, col, "high_income") for col in columns]
print(information_gains)
highest_gain_index = information_gains.index(max(information_gains))
highest_gain = columns[highest_gain_index]

def find_best_column(data, target_name, columns):
    # Fill in the logic here to automatically find the column in columns to split on
    # data is a dataframe
    # target_name is the name of the target variable
    # columns is a list of potential columns to split on
    orig_entropy = calc_entropy(data[target_name])
    information_gains = [ calc_information_gain(data, col, "high_income") for col in columns ]
    highest_gain_index = information_gains.index(max(information_gains))
    highest_gain = columns[highest_gain_index]

    return highest_gain

# A list of columns to potentially split income with
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]
income_split = find_best_column(income, "high_income", columns)
print(income_split)

# We'll use lists to store our labels for nodes (when we find them)
# Lists can be accessed inside our recursive function, whereas integers can't.
# Look at the python missions on scoping for more information on this topic
label_1s = []
label_0s = []


def id3(data, target, columns):
    # The pandas.unique method will return a list of all the unique values in a series
    unique_targets = pandas.unique(data[target])

    if len(unique_targets) == 1:
        # Insert code here to append 1 to label_1s or 0 to label_0s, based on what we should label the node
        # See lines 2 and 3 in the algorithm
        if unique_targets[0] == 1:
            label_1s.append(1)
        elif unique_targets[0] == 0:
            label_0s.append(0)

        # Returning here is critical -- if we don't, the recursive tree will never finish, and run forever
        # See our example above for when we returned
        return

    # Find the best column to split on in our data
    best_column = find_best_column(data, target, columns)
    # Find the median of the column
    column_median = data[best_column].median()


# Create a dictionary to hold the tree
# It has to be outside of the function so we can access it later
tree = {}

# This list will let us number the nodes
# It has to be a list so we can access it inside the function
nodes = []


def id3(data, target, columns, tree):
    unique_targets = pandas.unique(data[target])

    # Assign the number key to the node dictionary
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        # Insert code here that assigns the "label" field to the node dictionary
        tree["label"] = unique_targets[0]
        return

    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()

    # Insert code here that assigns the "column" and "median" fields to the node dictionary
    tree["column"] = best_column
    tree["median"] = column_median

    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]

    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])


# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"], tree)


def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))


def print_node(tree, depth):
    # Check for the presence of "label" in the tree
    if "label" in tree:
        # If found, then this is a leaf, so print it and return
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        # This is critical -- without it, you'll get infinite recursion
        return
    # Print information about what the node is splitting on
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)

    # Create a list of tree branches
    branches = [tree["left"], tree["right"]]

    # Insert code here to recursively call print_node on each branch
    # Don't forget to increment depth when you pass it in
    depth += 1
    print_node(branches[0], depth)
    print_node(branches[1], depth)


print_node(tree, 0)


def predict(tree, row):
    if "label" in tree:
        return tree["label"]

    column = tree["column"]
    median = tree["median"]

    # Insert code here to check whether row[column] is less than or equal to median
    # If it's less than or equal, return the result of predicting on the left branch of the tree
    # If it's greater, return the result of predicting on the right branch of the tree
    # Remember to use the return statement to return the result!
    if row[column] <= median:
        return predict(tree["left"], row)
    else:
        return predict(tree["right"], row)


# Print the prediction for the first row in our data
print(predict(tree, data.iloc[0]))

new_data = pandas.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
    ])
# Assign column names to the data
new_data.columns = ["age", "marital_status"]

def batch_predict(tree, df):
    # Insert your code here
    return df.apply(lambda x : predict(tree, x), axis=1)

predictions = batch_predict(tree, new_data)

