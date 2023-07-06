import csv

# Initialize an empty dictionary to hold the counts
counts = {}

# Open the CSV file
with open(r'model\sign_classifier\dataset\test_dataset.csv', 'r') as f:
    reader = csv.reader(f)

    # Loop over each row in the CSV file
    for row in reader:
        # Get the value of the first column
        first_column_value = int(row[0])

        # If this value is already a key in the dictionary, increment its count
        if first_column_value in counts:
            counts[first_column_value] += 1
        # Otherwise, add it to the dictionary with a count of 1
        else:
            counts[first_column_value] = 1

# Print the counts, ordered by key
for key in sorted(counts.keys()):
    print(f"Value: {key}, Count: {counts[key]}")
