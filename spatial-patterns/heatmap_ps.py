import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(csv_file):
    # Load the results from the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the column names in the CSV file match these
    # You might need to adjust these column names if they differ
    print(df.head())  # To check column names and structure

    # Pivot the DataFrame to get the format suitable for a heatmap
    heatmap_data = df.pivot(index='p', columns='sigma', values='integral')

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)

    # Set labels and title
    plt.xlabel('sigma value')
    plt.ylabel('p value')
    plt.title('Heatmap of Integration of Absolute Difference')
    
    # Save the heatmap
    plt.tight_layout()
    plt.savefig('spatial-patterns/heatmap-FCAT3_2.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
plot_heatmap('spatial-patterns/results_FCAT11_2.csv')
