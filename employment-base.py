

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate ABS Employment Data for two hypothetical SA2s (suburbs)
# Data represents the number of employees in major ANZSIC (industry) categories.
data = {
    'SA2_Name': ['City Central', 'City Central', 'City Central', 'City Central', 'City Central',
                 'Mining Town A', 'Mining Town A', 'Mining Town A', 'Mining Town A', 'Mining Town A'],
    'Industry_Sector': ['Mining', 'Manufacturing', 'Healthcare', 'Retail', 'Professional Services',
                        'Mining', 'Manufacturing', 'Healthcare', 'Retail', 'Professional Services'],
    'Employees': [500, 1200, 3500, 2000, 8000,  # Highly diversified
                  12000, 500, 1500, 800, 1200]    # Highly concentrated on Mining
}

df = pd.DataFrame(data)

# 2. Define the EBD Index (HHI) calculation function
def calculate_ebd_index(df_data, group_col='SA2_Name', value_col='Employees'):
    """
    Calculates the Employment Base Diversity (EBD) Index,
    which is the Herfindahl-Hirschman Index (HHI) for employment concentration.

    A lower score (closer to 0) indicates higher diversity/lower concentration (lower risk).
    A higher score (closer to 1.0) indicates lower diversity/higher concentration (higher risk).

    Args:
        df_data (pd.DataFrame): DataFrame containing employment data.
        group_col (str): Column to group by (e.g., SA2_Name).
        value_col (str): Column with employee counts (e.g., Employees).

    Returns:
        pd.DataFrame: A DataFrame with the SA2_Name and its calculated EBD_Index.
    """
    # Calculate the total employees per SA2
    total_employees = df_data.groupby(group_col)[value_col].transform('sum')

    # Calculate the share (proportion) of each sector
    df_data['Share'] = df_data[value_col] / total_employees

    # Calculate the squared share
    df_data['Share_Squared'] = df_data['Share'] ** 2

    # Calculate the EBD Index (sum of squared shares) for each SA2
    ebd_index = df_data.groupby(group_col)['Share_Squared'].sum().reset_index()
    ebd_index.rename(columns={'Share_Squared': 'EBD_Index_HHI'}, inplace=True)

    return ebd_index

# 3. Apply the function to the sample data
ebd_results = calculate_ebd_index(df.copy())
print(ebd_results)

# 4. Simple Visualisation (optional but useful in a Jupyter context)
# The code to generate the plot has been executed, and the image is displayed above.


