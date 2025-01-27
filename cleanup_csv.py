import pandas as pd

def main():
    # Step 1: Load and clean the original CSV file
    dataset_years_available = ["1995", "2005", "2015", "2020"]

    Year_input = input("Which year [1995, 2005, 2015, 2020] of the dataset would you like to clean? (exit to quit program): ")

    while(Year_input not in dataset_years_available):
        if Year_input == "exit":
            return
        Year_input = input("Which year [1995, 2005, 2015, 2020] of the dataset would you like to clean? (exit to quit program): ")


    file_path = f'data/BACI_HS92_Y{Year_input}_V202401b.csv'
    df = pd.read_csv(file_path)

    # Rename columns for better readability
    df = df.rename(columns={'t': 'Year', 'i': 'Exporter', 'j': 'Importer', 'k': 'Product', 'v': 'Value', 'q': 'Quantity'})

    # Group by Exporter and Importer, summing the Value column
    df_merged = df.groupby(["Exporter", "Importer"], as_index=False).agg({
        "Product": 'first',  # Keep the first entry for the product
        "Value": 'sum'      # Sum the values
    })


    # Drop unnecessary columns
    df_merged.drop(columns=["Product"], inplace=True)  # Drop the Product column

    # Save the cleaned data
    df_merged.to_csv(f'cleaned_{Year_input}.csv', index=False)
    print(f"Cleaning complete. Cleaned file saved as: cleaned_{Year_input}.csv")

    # Step 2: Map country codes to country names
    country_code_df = pd.read_csv('data/country_codes_V202401b.csv')

    # Create a mapping dictionary
    code_to_name = dict(zip(country_code_df['country_code'], country_code_df['country_name']))

    # Replace Exporter and Importer codes with country names
    df_merged['Exporter'] = df_merged['Exporter'].map(code_to_name)
    df_merged['Importer'] = df_merged['Importer'].map(code_to_name)

    # Save the final transformed data
    df_merged.to_csv(f'renamed_{Year_input}.csv', index=False)
    print(f"CSV transformation complete! Transformed file saved as: renamed_{Year_input}.csv")

if __name__ == "__main__":
    main()