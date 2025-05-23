# Import the pandas library for data manipulation
import pandas as pd

# Import the os module to interact with the operating system (used for file renaming)
import os

# Notify user that the CSV cleaning process has started
print("Cleaning the CSV file...")

# Try to read the CSV file using various encodings to handle possible encoding errors
try:
    # Attempt to read the CSV file using the default encoding
    data = pd.read_csv('Placement_data_full_class.csv')
    print("CSV file read successfully with default encoding")
except Exception as e:
    # If default encoding fails, print the error
    print(f"Error with default encoding: {str(e)}")
    try:
        # Try reading the file with UTF-8 encoding
        data = pd.read_csv('Placement_data_full_class.csv', encoding='utf-8')
        print("CSV file read successfully with utf-8 encoding")
    except Exception as e:
        # If UTF-8 fails, print the error
        print(f"Error with utf-8 encoding: {str(e)}")
        try:
            # Try reading the file with Latin-1 encoding
            data = pd.read_csv('Placement_data_full_class.csv', encoding='latin-1')
            print("CSV file read successfully with latin-1 encoding")
        except Exception as e:
            # If all attempts fail, print error and exit the script
            print(f"Error with latin-1 encoding: {str(e)}")
            print("Could not read the CSV file with common encodings.")
            exit(1)

# If the file is successfully read, display its shape (rows, columns)
print(f"Original data shape: {data.shape}")

# Print the column names of the dataset
print(f"Original columns: {data.columns.tolist()}")

# Rename the original CSV file to create a backup
os.rename('Placement_data_full_class.csv', 'Placement_data_full_class_original.csv')
print("Created backup of original CSV file")

# Save the cleaned version of the data back to a new CSV file
data.to_csv('Placement_data_full_class.csv', index=False)
print("Saved cleaned CSV file")

# Try to verify that the cleaned file was saved and can be read properly
try:
    # Read the cleaned file
    cleaned_data = pd.read_csv('Placement_data_full_class.csv')
    
    # Print shape of the cleaned dataset
    print(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Print column names of the cleaned dataset
    print(f"Cleaned columns: {cleaned_data.columns.tolist()}")
    
    # Show the first 5 rows for verification
    print("First 5 rows of cleaned data:")
    print(cleaned_data.head())
    
    # Print data types of each column
    print("\nData types:")
    print(cleaned_data.dtypes)
    
    # Print number of missing values per column
    print("\nMissing values:")
    print(cleaned_data.isnull().sum())
    
    # Indicate success
    print("CSV file cleaned and verified successfully!")
except Exception as e:
    # If any error occurs during verification, print the error
    print(f"Error verifying cleaned CSV file: {str(e)}")
    
    # Restore the original CSV file from backup
    os.rename('Placement_data_full_class_original.csv', 'Placement_data_full_class.csv')
    print("Restored original CSV file due to verification failure")
