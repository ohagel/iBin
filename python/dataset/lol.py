def subtract_one_from_third_column(file_path):
    # Read the file and modify content
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read all lines

    modified_lines = []
    for line in lines:
        elements = line.strip().split(',')  # Split the line by comma
        if len(elements) >= 3:
            third_column = int(elements[2]) - 1  # Subtract 1 from the third column
            elements[2] = str(third_column)  # Convert back to string
        modified_lines.append(','.join(elements))  # Join elements with a comma

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write('\n'.join(modified_lines))

if __name__ == "__main__":
    # Provide the path to your labels.txt file
    file_path = 'labels.txt'

    # Call the function to subtract 1 from the third column
    subtract_one_from_third_column(file_path)
    print(f"Subtracted 1 from the third column in '{file_path}'")