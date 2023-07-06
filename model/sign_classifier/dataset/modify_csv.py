import csv
import random

# Initialize an empty dictionary to hold the rows
rows = {}
counts = {}

# Open the CSV file
with open(r'model\sign_classifier\dataset\sign_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    # Loop over each row in the CSV file
    for row in reader:
        # Get the value of the first column
        first_column_value = row[0]

        # If this value is already a key in the dictionary, increment its count
        # and add the row to the list of rows for this value
        if first_column_value in counts:
            counts[first_column_value] += 1
            rows[first_column_value].append(row)
        # Otherwise, add it to the dictionary with a count of 1 and a new list containing this row
        else:
            counts[first_column_value] = 1
            rows[first_column_value] = [row]

# For each key in the dictionary, if the count is greater than 3000, randomly remove rows until only 3000 remain
for key in counts.keys():
    if counts[key] > 3000:
        to_remove = counts[key] - 3000
        for _ in range(to_remove):
            row_to_remove = random.choice(rows[key])
            rows[key].remove(row_to_remove)

# Now rows contains at most 3000 rows for each first column value
# You can write this back to a new CSV file if desired
with open(r'model\sign_classifier\dataset\sign_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for key in rows.keys():
        for row in rows[key]:
            writer.writerow(row)
