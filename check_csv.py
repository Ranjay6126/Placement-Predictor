# Import the pandas library for data manipulation and analysis
import pandas as pd 

# Try block to handle potential errors while reading the CSV file
try:
    # Attempt to read the CSV file named 'Placement_data_full_class.csv' into a pandas DataFrame
    data = pd.read_csv('Placement_data_full_class.csv')
    
    # Print a success message
    print("CSV file read successfully!")
    
    # Print the shape of the dataset (rows, columns)
    print("\nDataset shape:", data.shape)
    
    # Print the list of column names in the dataset
    print("\nColumn names:", data.columns.tolist())
    
    # Display the first 5 rows of the dataset for preview
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Print the data types of each column
    print("\nData types:")
    print(data.dtypes)
    
    # Check and print the number of missing values in each column
    print("\nMissing values:")
    print(data.isnull().sum())

# Catch any exception that occurs while reading the CSV and print the error message
except Exception as e:
    print(f"Error reading CSV file: {str(e)}")
