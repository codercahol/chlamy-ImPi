import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

def get_variance_for_genes(data) :
    # Group by 'mutated_gene' and 'light_regime'
    # grouped = data.groupby(['mutated_genes', 'light_regime'])
    grouped = data.groupby(['mutant_ID', 'light_regime'])

    # Initialize a dictionary to store the variances and sample counts
    variances_y2 = {}
    variances_ynpq = {}
    sample_counts = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in grouped:
        # Calculate the variance for each 'y2_i' column within the group
        group_variances_y2 = group.filter(regex=r'^y2_\d+$').dropna(axis=1).values.var()
        group_variances_ynpq = group.filter(regex=r'^ynpq_\d+$').dropna(axis=1).values.var()
        if len(group) > 1 :
            # Compute the mean variance across all 'y2_i' columns for the gene within the light regime
            mean_variance_y2 = group_variances_y2
            mean_variance_ynpq = group_variances_ynpq
            # Store the mean variance in the dictionary
            # Convert tuple to string for consistent key format
            key = '#'.join(map(str, name))
            variances_y2[key] = mean_variance_y2
            variances_ynpq[key] = mean_variance_ynpq
            # Count the number of samples for the gene within the light regime
            sample_counts[key] = len(group)

    # Convert the dictionaries to DataFrames
    # variance_df = pd.DataFrame(variances.items(), columns=['mutated_genes_light_regime', 'mean_variance'])
    # sample_counts_df = pd.DataFrame(sample_counts.items(), columns=['mutated_genes_light_regime', 'sample_count'])
    variance_y2_df = pd.DataFrame(variances_y2.items(), columns=['mutant_ID_light_regime', 'mean_variance_y2'])
    variance_ynpq_df = pd.DataFrame(variances_ynpq.items(), columns=['mutant_ID_light_regime', 'mean_variance_ynpq'])
    sample_counts_df = pd.DataFrame(sample_counts.items(), columns=['mutant_ID_light_regime', 'sample_count'])

    # Merge the two DataFrames on the 'mutated_genes_light_regime' column
    result_df = variance_y2_df.merge(variance_ynpq_df, on='mutant_ID_light_regime')
    result_df = result_df.merge(sample_counts_df, on='mutant_ID_light_regime')

    # Split the 'mutated_genes_light_regime' column into 'mutated_genes' and 'light_regime'
    result_df[['mutant_ID', 'light_regime']] = result_df['mutant_ID_light_regime'].str.split('#', n=1, expand=True)

    # Drop the intermediate column
    result_df.drop(columns=['mutant_ID_light_regime'], inplace=True)

    return result_df

def get_mean_intra_distance_for_WT(data) :
    from scipy.spatial.distance import pdist, squareform

    # Group by 'mutated_gene' and 'light_regime'
    grouped = data.groupby(['light_regime', 'plate'])

    # Initialize a dictionary to store the intra_gene_distances and sample counts
    intra_gene_distances_y2 = {}
    intra_gene_distances_var_y2 = {}
    intra_gene_distances_ynpq = {}
    intra_gene_distances_var_ynpq = {}
    sample_counts = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in grouped:
        # Calculate the variance for each 'y2_i' column within the group
        distances_y2 = pdist(group.filter(like='y2_').dropna(axis=1).values, metric=distance_metric)
        distances_ynpq = pdist(group.filter(like='ynpq_').dropna(axis=1).values, metric=distance_metric)
        if len(group) > 1 :
            mean_distance_y2 = distances_y2.mean()
            var_distance_y2 = distances_y2.var()
            mean_distance_ynpq = distances_ynpq.mean()
            var_distance_ynpq = distances_ynpq.var()
            # Store the mean distance in the dictionary
            # Convert tuple to string for consistent key format
            key = '#'.join(map(str, name))
            intra_gene_distances_y2[key] = mean_distance_y2
            intra_gene_distances_var_y2[key] = var_distance_y2
            intra_gene_distances_ynpq[key] = mean_distance_ynpq
            intra_gene_distances_var_ynpq[key] = var_distance_ynpq
            # Count the number of samples for the gene within the light regime
            sample_counts[key] = len(group)

    # Convert the dictionaries to DataFrames
    intra_gene_distances_df_y2 = pd.DataFrame(intra_gene_distances_y2.items(), columns=['light_regime_plate', 'mean_intra_gene_distance_y2'])
    intra_gene_distances_df_var_y2 = pd.DataFrame(intra_gene_distances_var_y2.items(), columns=['light_regime_plate', 'var_intra_gene_distance_y2'])
    intra_gene_distances_df_ynpq = pd.DataFrame(intra_gene_distances_ynpq.items(), columns=['light_regime_plate', 'mean_intra_gene_distance_ynpq'])
    intra_gene_distances_df_var_ynpq = pd.DataFrame(intra_gene_distances_var_ynpq.items(), columns=['light_regime_plate', 'var_intra_gene_distance_ynpq'])
    sample_counts_df = pd.DataFrame(sample_counts.items(), columns=['light_regime_plate', 'sample_count'])

    # Merge the two DataFrames on the 'mutated_genes_light_regime' column
    intra_gene_distances_df = intra_gene_distances_df_y2.merge(intra_gene_distances_df_var_y2, on='light_regime_plate')
    intra_gene_distances_df = intra_gene_distances_df.merge(intra_gene_distances_df_ynpq, on='light_regime_plate')
    intra_gene_distances_df = intra_gene_distances_df.merge(intra_gene_distances_df_var_ynpq, on='light_regime_plate')
    intra_gene_distances_df = intra_gene_distances_df.merge(sample_counts_df, on='light_regime_plate')

    # Split the 'mutated_genes_light_regime' column into 'mutated_genes' and 'light_regime'
    intra_gene_distances_df[['light_regime', 'plate']] = intra_gene_distances_df['light_regime_plate'].str.split('#', n=1, expand=True)

    # Drop the intermediate column
    intra_gene_distances_df.drop(columns=['light_regime_plate'], inplace=True)

    return intra_gene_distances_df

def get_intra_distance_for_WT(data) :
    from scipy.spatial.distance import pdist, squareform

    # Group by 'mutated_gene' and 'light_regime'
    grouped = data.groupby(['light_regime', 'plate'])

    # Initialize a dictionary to store the intra_gene_distances and sample counts
    intra_gene_distances_y2 = {}
    intra_gene_distances_ynpq = {}
    sample_counts = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in grouped:
        # Calculate the variance for each 'y2_i' column within the group
        distances_y2 = pdist(group.filter(like='y2_').dropna(axis=1).values, metric=distance_metric)
        distances_ynpq = pdist(group.filter(like='ynpq_').dropna(axis=1).values, metric=distance_metric)
        if len(group) > 1 :
            # Store the mean distance in the dictionary
            # Convert tuple to string for consistent key format
            key = '#'.join(map(str, name))
            intra_gene_distances_y2[key] = distances_y2
            intra_gene_distances_ynpq[key] = distances_ynpq

    # Convert the dictionaries to DataFrames
    intra_gene_distances_df_y2 = pd.DataFrame(intra_gene_distances_y2.items(), columns=['light_regime_plate', 'pairwise_distances_y2'])
    intra_gene_distances_df_ynpq = pd.DataFrame(intra_gene_distances_ynpq.items(), columns=['light_regime_plate', 'pairwise_distances_ynpq'])

    # Merge the two DataFrames on the 'mutated_genes_light_regime' column
    intra_gene_distances_df = intra_gene_distances_df_y2.merge(intra_gene_distances_df_ynpq, on='light_regime_plate')

    # Split the 'mutated_genes_light_regime' column into 'mutated_genes' and 'light_regime'
    intra_gene_distances_df[['light_regime', 'plate']] = intra_gene_distances_df['light_regime_plate'].str.split('#', n=1, expand=True)

    # Drop the intermediate column
    intra_gene_distances_df.drop(columns=['light_regime_plate'], inplace=True)

    return intra_gene_distances_df

def get_mean_intra_distance(data):
    # Group by 'mutated_gene' and 'light_regime'
    grouped = data.groupby(['light_regime'])

    # Initialize a dictionary to store the intra_gene_distances and sample counts
    intra_distances_y2 = {}
    intra_distances_var_y2 = {}
    intra_distances_ynpq = {}
    intra_distances_var_ynpq = {}
    sample_counts = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in grouped:
        # Calculate the variance for each 'y2_i' column within the group
        if name[0] == '20h_ML' or name[0] == '20h_HL' or name[0] == '2h-2h':
            distances_y2 = pdist(group.iloc[:2000].filter(like='y2_').iloc[:, :40].values, metric=distance_metric)
            distances_ynpq = pdist(group.iloc[:2000].filter(like='ynpq_').iloc[:, :40].values, metric=distance_metric)
        else :
            distances_y2 = pdist(group.iloc[:2000].filter(like='y2_').values, metric=distance_metric)
            distances_ynpq = pdist(group.iloc[:2000].filter(like='ynpq_').values, metric=distance_metric)
        if len(group) > 1 :
            mean_distance_y2 = distances_y2.mean()
            var_distance_y2 = distances_y2.var()
            mean_distance_ynpq = distances_ynpq.mean()
            var_distance_ynpq = distances_ynpq.var()
            # Store the mean distance in the dictionary
            # Convert tuple to string for consistent key format
            key = '#'.join(map(str, name))
            intra_distances_y2[key] = mean_distance_y2
            intra_distances_var_y2[key] = var_distance_y2
            intra_distances_ynpq[key] = mean_distance_ynpq
            intra_distances_var_ynpq[key] = var_distance_ynpq
            # Count the number of samples for the gene within the light regime
            sample_counts[key] = len(group)

    # Convert the dictionaries to DataFrames
    intra_distances_df_y2 = pd.DataFrame(intra_distances_y2.items(), columns=['light_regime', 'mean_distance_y2'])
    intra_distances_df_var_y2 = pd.DataFrame(intra_distances_var_y2.items(), columns=['light_regime', 'var_distance_y2'])
    intra_distances_df_ynpq = pd.DataFrame(intra_distances_ynpq.items(), columns=['light_regime', 'mean_distance_ynpq'])
    intra_distances_df_var_ynpq = pd.DataFrame(intra_distances_var_ynpq.items(), columns=['light_regime', 'var_distance_ynpq'])
    sample_counts_df = pd.DataFrame(sample_counts.items(), columns=['light_regime', 'sample_count'])

    # Merge the two DataFrames on the 'mutated_genes_light_regime' column
    intra_distances_df = intra_distances_df_y2.merge(intra_distances_df_var_y2, on='light_regime')
    intra_distances_df = intra_distances_df.merge(intra_distances_df_ynpq, on='light_regime')
    intra_distances_df = intra_distances_df.merge(intra_distances_df_var_ynpq, on='light_regime')
    intra_distances_df = intra_distances_df.merge(sample_counts_df, on='light_regime')

    # Split the 'mutated_genes_light_regime' column into 'mutated_genes' and 'light_regime'
    intra_distances_df['light_regime'] = intra_distances_df['light_regime'].str.split('#', n=1, expand=True)

    return intra_distances_df

def get_mean_intra_distance_for_genes(data) :
    from scipy.spatial.distance import pdist, squareform

    # Group by 'mutated_gene' and 'light_regime'
    grouped = data.groupby(['mutated_genes', 'light_regime'])

    # Initialize a dictionary to store the intra_gene_distances and sample counts
    intra_gene_distances_y2 = {}
    intra_gene_distances_var_y2 = {}
    intra_gene_distances_ynpq = {}
    intra_gene_distances_var_ynpq = {}
    sample_counts = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in grouped:
        # Calculate the variance for each 'y2_i' column within the group
        if name[1] == '20h_ML' or name[1] == '20h_HL' or name[1] == '2h-2h':
            distances_y2 = pdist(group.filter(like='y2_').iloc[:, :40].values, metric=distance_metric)
            distances_ynpq = pdist(group.filter(like='ynpq_').iloc[:, :40].values, metric=distance_metric)
        else :
            distances_y2 = pdist(group.filter(like='y2_').values, metric=distance_metric)
            distances_ynpq = pdist(group.filter(like='ynpq_').values, metric=distance_metric)
        if len(group) > 1 :
            mean_distance_y2 = distances_y2.mean()
            var_distance_y2 = distances_y2.var()
            mean_distance_ynpq = distances_ynpq.mean()
            var_distance_ynpq = distances_ynpq.var()
            # Store the mean distance in the dictionary
            # Convert tuple to string for consistent key format
            key = '#'.join(map(str, name))
            intra_gene_distances_y2[key] = mean_distance_y2
            intra_gene_distances_var_y2[key] = var_distance_y2
            intra_gene_distances_ynpq[key] = mean_distance_ynpq
            intra_gene_distances_var_ynpq[key] = var_distance_ynpq
            # Count the number of samples for the gene within the light regime
            sample_counts[key] = len(group)

    # Convert the dictionaries to DataFrames
    intra_gene_distances_df_y2 = pd.DataFrame(intra_gene_distances_y2.items(), columns=['mutated_genes_light_regime', 'mean_intra_gene_distance_y2'])
    intra_gene_distances_df_var_y2 = pd.DataFrame(intra_gene_distances_var_y2.items(), columns=['mutated_genes_light_regime', 'var_intra_gene_distance_y2'])
    intra_gene_distances_df_ynpq = pd.DataFrame(intra_gene_distances_ynpq.items(), columns=['mutated_genes_light_regime', 'mean_intra_gene_distance_ynpq'])
    intra_gene_distances_df_var_ynpq = pd.DataFrame(intra_gene_distances_var_ynpq.items(), columns=['mutated_genes_light_regime', 'var_intra_gene_distance_ynpq'])
    sample_counts_df = pd.DataFrame(sample_counts.items(), columns=['mutated_genes_light_regime', 'sample_count'])

    # Merge the two DataFrames on the 'mutated_genes_light_regime' column
    intra_gene_distances_df = intra_gene_distances_df_y2.merge(intra_gene_distances_df_var_y2, on='mutated_genes_light_regime')
    intra_gene_distances_df = intra_gene_distances_df.merge(intra_gene_distances_df_ynpq, on='mutated_genes_light_regime')
    intra_gene_distances_df = intra_gene_distances_df.merge(intra_gene_distances_df_var_ynpq, on='mutated_genes_light_regime')
    intra_gene_distances_df = intra_gene_distances_df.merge(sample_counts_df, on='mutated_genes_light_regime')

    # Split the 'mutated_genes_light_regime' column into 'mutated_genes' and 'light_regime'
    intra_gene_distances_df[['mutated_genes', 'light_regime']] = intra_gene_distances_df['mutated_genes_light_regime'].str.split('#', n=1, expand=True)

    # Drop the intermediate column
    intra_gene_distances_df.drop(columns=['mutated_genes_light_regime'], inplace=True)

    return intra_gene_distances_df

def get_avg_pairwise_distances_for_feature_combinations(data):
    # Group by 'mutated_genes' and 'light_regime'
    grouped = data.groupby(['mutated_genes', 'light_regime'])

    # Initialize dictionaries to store distances and sample counts
    intra_feature_distances_y2 = {}
    intra_feature_distances_ynpq = {}
    sample_counts = {}

    # Iterate over each group
    for name, group in grouped:
        features = group['feature'].values
        if None not in features:
            unique_features = np.unique(features)
            
            # Calculate pairwise distances for 'y2_' and 'ynpq_' columns within the group
            if name[1] in ['20h_ML', '20h_HL', '2h-2h']:
                distances_y2 = squareform(pdist(group.filter(like='y2_').iloc[:, :40].values, metric=distance_metric))
                distances_ynpq = squareform(pdist(group.filter(like='ynpq_').iloc[:, :40].values, metric=distance_metric))
            else:
                distances_y2 = squareform(pdist(group.filter(like='y2_').values, metric=distance_metric))
                distances_ynpq = squareform(pdist(group.filter(like='ynpq_').values, metric=distance_metric))
            
            # Create a DataFrame to store the distances along with feature information
            distances_df_y2 = pd.DataFrame(distances_y2, index=features, columns=features)
            distances_df_ynpq = pd.DataFrame(distances_ynpq, index=features, columns=features)
            
            for i in range(len(unique_features)):
                for j in range(i+1):
                    f1 = unique_features[i]
                    f2 = unique_features[j]
                    key = f'{name[0]}#{name[1]}#{f1}#{f2}'
                    relevant_distances_y2 = distances_df_y2.loc[f1, f2]
                    relevant_distances_ynpq = distances_df_ynpq.loc[f1, f2]

                    if type(relevant_distances_y2) == np.float64:
                        relevant_distances_y2 = np.array([relevant_distances_y2])
                    if type(relevant_distances_ynpq) == np.float64:
                        relevant_distances_ynpq = np.array([relevant_distances_ynpq])
                    
                    # Exclude the zero distances (self-distances)
                    if f1 == f2:
                        relevant_distances_y2 = relevant_distances_y2[relevant_distances_y2 != 0]
                        relevant_distances_ynpq = relevant_distances_ynpq[relevant_distances_ynpq != 0]
                    
                    if len(relevant_distances_y2) > 0:
                        avg_distance_y2 = np.mean(relevant_distances_y2)
                    else:
                        avg_distance_y2 = np.nan  # or some other placeholder

                    if len(relevant_distances_ynpq) > 0:
                        avg_distance_ynpq = np.mean(relevant_distances_ynpq)
                    else:
                        avg_distance_ynpq = np.nan
                    
                    intra_feature_distances_y2[key] = avg_distance_y2
                    intra_feature_distances_ynpq[key] = avg_distance_ynpq
                    sample_counts[key] = group.shape[0]
    
    # Convert dictionaries to DataFrames
    intra_feature_distances_df_y2 = pd.DataFrame(intra_feature_distances_y2.items(), columns=['mutated_genes_light_regime_feature_combination', 'avg_inter_feature_distance_y2'])
    intra_feature_distances_df_ynpq = pd.DataFrame(intra_feature_distances_ynpq.items(), columns=['mutated_genes_light_regime_feature_combination', 'avg_inter_feature_distance_ynpq'])
    sample_counts_df = pd.DataFrame(sample_counts.items(), columns=['mutated_genes_light_regime_feature_combination', 'sample_count'])

    # Merge DataFrames
    intra_feature_distances_df = intra_feature_distances_df_y2.merge(intra_feature_distances_df_ynpq, on='mutated_genes_light_regime_feature_combination')
    intra_feature_distances_df = intra_feature_distances_df.merge(sample_counts_df, on='mutated_genes_light_regime_feature_combination')

    # Split the 'mutated_genes_light_regime_feature_combination' column
    intra_feature_distances_df[['mutated_genes', 'light_regime', 'feature1', 'feature2']] = intra_feature_distances_df['mutated_genes_light_regime_feature_combination'].str.split('#', expand=True)

    # Drop the intermediate column
    intra_feature_distances_df.drop(columns=['mutated_genes_light_regime_feature_combination'], inplace=True)

    return intra_feature_distances_df

def get_mean_var_WT(result_df, type='mean_variance') :
    # Get the mean variance of the WT genes
    result_df_WT = result_df[result_df['mutated_genes'] == '']
    mean_var_WT_y2 = result_df_WT[type + '_y2'].mean()
    mean_var_WT_ynpq = result_df_WT[type + '_ynpq'].mean()

    return mean_var_WT_y2, mean_var_WT_ynpq

def plot_genes_self_similarity(result_df, type='mean_variance') :
    mean_var_WT_y2, mean_var_WT_ynpq = get_mean_var_WT(result_df, type=type)
    result_df_not_WT = result_df[result_df['mutated_genes'] != '']

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot the first subplot
    ax1.scatter(result_df_not_WT['sample_count'], result_df_not_WT[type + '_y2'], marker='.')
    ax1.axhline(y=mean_var_WT_y2, color='r', linestyle='--')
    ax1.set_xlabel('Sample Count')
    ax1.set_ylabel(type)
    ax1.set_title(type + ' Y(II) vs. Sample Count')

    # Plot the second subplot
    ax2.scatter(result_df_not_WT['sample_count'], result_df_not_WT[type + '_ynpq'], marker='.')
    ax2.axhline(y=mean_var_WT_ynpq, color='r', linestyle='--')
    ax2.set_xlabel('Sample Count')
    ax2.set_ylabel(type)
    ax2.set_title(type + ' Y(NPQ) vs. Sample Count')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    # plot 1/sqrt(sample_count) as a function of the mean_variance
    # I am not sure of this plot, the question is how to compensate the variance with the number of samples, in other words, how fast the variance converges to the true variance with the number of samples

    # plt.scatter(result_df_not_WT['sample_count'], 0.02/np.sqrt(result_df_not_WT['sample_count'])) 


def distance_metric(series1, series2):
    return ((series1 - series2) ** 2).sum()

# Calculate pairwise distances between all time-series within each group
def calculate_distances_var(group):
    y2_values = group.filter(like='y2_').dropna(axis=1).values
    ynpq_values = group.filter(like='ynpq_').dropna(axis=1).values
    distances_y2 = pdist(y2_values, metric=distance_metric)
    distances_ynpq = pdist(ynpq_values, metric=distance_metric)
    return squareform(distances_y2), squareform(distances_ynpq), y2_values.var(axis=1)


# Check if at least 2/3 of the time-series are within the threshold distance
def flag_time_series_mutant(group, name, threshold_distance_y2, threshold_distance_ynpq, p=(2/3), threshold_variance = 1):
    distances_y2, distances_ynpq, var = calculate_distances_var(group)
    num_series = len(group)
    ok_indices_y2 = []
    ok_indices_ynpq = []
    flags_y2 = group['flag_y2'].values
    flags_ynpq = group['flag_ynpq'].values
    if name[0] != '' :
        for i in range(num_series):
            close_count_y2 = sum(1 for d in distances_y2[i] if d < threshold_distance_y2[name[1]])
            close_count_ynpq = sum(1 for d in distances_ynpq[i] if d < threshold_distance_ynpq[name[1]])
            if close_count_y2 >= p*num_series and var[i] < threshold_variance:
                ok_indices_y2.append(i)
            if close_count_ynpq >= p*num_series and var[i] < threshold_variance:
                ok_indices_ynpq.append(i)

        flags_y2 = ['ok' if i in ok_indices_y2 else 'weird' for i in range(num_series)]
        flags_ynpq = ['ok' if i in ok_indices_ynpq else 'weird' for i in range(num_series)]

    return pd.DataFrame(flags_y2, index=group.index, columns=['flag_y2']), pd.DataFrame(flags_ynpq, index=group.index, columns=['flag_ynpq'])

def flag_time_series_WT(group, name, threshold_distance_y2, threshold_distance_ynpq, p=(2/3), threshold_variance = 1):
    num_series = len(group)
    flags_y2 = [np.nan for i in range(num_series)]
    flags_ynpq = [np.nan for i in range(num_series)]
    if name[0] == '' :
        distances_y2, distances_ynpq, var = calculate_distances_var(group)
        ok_indices_y2 = []
        ok_indices_ynpq = []

        for i in range(num_series):
            close_count_y2 = sum(1 for d in distances_y2[i] if d < threshold_distance_y2[name[2]])
            close_count_ynpq = sum(1 for d in distances_ynpq[i] if d < threshold_distance_ynpq[name[2]])
            if close_count_y2 >= p*num_series and var[i] < threshold_variance:
                ok_indices_y2.append(i)
            if close_count_ynpq >= p*num_series and var[i] < threshold_variance:
                ok_indices_ynpq.append(i)
        flags_y2 = ['ok' if i in ok_indices_y2 else 'weird' for i in range(num_series)]
        flags_ynpq = ['ok' if i in ok_indices_ynpq else 'weird' for i in range(num_series)]
    
    return pd.DataFrame(flags_y2, index=group.index, columns=['flag_y2']), pd.DataFrame(flags_ynpq, index=group.index, columns=['flag_ynpq'])

def apply_flagging_mutants(data, threshold_distance_y2, threshold_distance_ynpq, p=(2/3), threshold_variance = 1):
    # Apply the flagging function to each group and concatenate the results
    flagged_series_y2 = pd.concat([flag_time_series_mutant(group, name, threshold_distance_y2, threshold_distance_ynpq, p, threshold_variance)[0] for name, group in data.groupby(['mutated_genes', 'light_regime'])])
    flagged_series_ynpq = pd.concat([flag_time_series_mutant(group, name, threshold_distance_y2, threshold_distance_ynpq, p, threshold_variance)[1] for name, group in data.groupby(['mutated_genes', 'light_regime'])])

    # Merge the two flagged series back to the original DataFrame
    data_copy = data.merge(flagged_series_y2, left_index=True, right_index=True, how='left')
    data_copy = data_copy.merge(flagged_series_ynpq, left_index=True, right_index=True, how='left')

    return data_copy

def apply_flagging_WT(data, threshold_distance_y2, threshold_distance_ynpq, p=(2/3), threshold_variance = 1):
    # Apply the flagging function to each group and concatenate the results
    flagged_series_y2 = pd.concat([flag_time_series_WT(group, name, threshold_distance_y2, threshold_distance_ynpq, p, threshold_variance)[0] for name, group in data.groupby(['mutated_genes', 'plate', 'light_regime'])])
    flagged_series_ynpq = pd.concat([flag_time_series_WT(group, name, threshold_distance_y2, threshold_distance_ynpq, p, threshold_variance)[1] for name, group in data.groupby(['mutated_genes', 'plate', 'light_regime'])])

    # Merge the two flagged series back to the original DataFrame
    data_copy = data.merge(flagged_series_y2, left_index=True, right_index=True, how='left')
    data_copy = data_copy.merge(flagged_series_ynpq, left_index=True, right_index=True, how='left')

    return data_copy

# Check if at least 2/3 of the time-series are within the threshold distance
def flag_time_series_2(name, group, alpha = 0.05, threshold_variance = 1):
    measurement = name[1]
    threshold_distance = chi2.ppf(1 - alpha, ((1/2)*group['num_frames'].values[0] - 2))
    distances, var = calculate_distances_var(group, measurement)
    distances = (1/(2*group['sigma'].values[0]**2))*distances
    num_series = len(group)
    ok_indices = []

    for i in range(num_series):
        close_count = sum(1 for d in distances[i] if d < threshold_distance)
        if close_count >= (2/3)*num_series and var[i] < threshold_variance:
            ok_indices.append(i)

    flags = ['ok' if i in ok_indices else 'weird' for i in range(num_series)]
    return pd.DataFrame(flags, index=group.index, columns=['flag'])

def apply_flagging_2(data, alpha = 0.05, threshold_variance = 1):
    # Apply the flagging function to each group and concatenate the results
    flagged_series = pd.concat([flag_time_series_2(name, group, alpha, threshold_variance) for name, group in data.groupby(['mutated_genes', 'light_regime'])])

    # Merge the flagged series back to the original DataFrame
    data_copy = data.merge(flagged_series, left_index=True, right_index=True, how='left')

    return data_copy

def calculate_shape_distances(group):
    y2_values = group.filter(regex=r'^y2_\d+$').dropna(axis=1).values
    y2_derivatives = np.gradient(y2_values, axis=1)
    ynpq_values = group.filter(regex=r'^ynpq_\d+$').dropna(axis=1).values
    ynpq_derivatives = np.gradient(ynpq_values, axis=1)
    return dtw.distance_matrix(y2_derivatives), dtw.distance_matrix(ynpq_derivatives)

def flag_time_series_shape_mutant(group, name, threshold_distances_y2, threshold_distances_ynpq, p=(2/3), threshold_variance = 1):
    num_series = len(group)
    ok_indices_y2 = []
    ok_indices_ynpq = []
    flags_y2 = group['flag_shape_y2'].values
    flags_ynpq = group['flag_shape_ynpq'].values
    if name[0] != '':
        distances_shape_y2, distances_shape_ynpq = calculate_shape_distances(group)
        threshold_distance_y2 = threshold_distances_y2[name[1]]
        threshold_distance_ynpq = threshold_distances_ynpq[name[1]]
        for i in range(num_series):
            close_count_y2 = sum(1 for d in distances_shape_y2[i] if d < threshold_distance_y2)
            close_count_ynpq = sum(1 for d in distances_shape_ynpq[i] if d < threshold_distance_ynpq)
            if close_count_y2 >= p*num_series:
                ok_indices_y2.append(i)
            if close_count_ynpq >= p*num_series:
                ok_indices_ynpq.append(i)

        flags_y2 = [False if i in ok_indices_y2 else True for i in range(num_series)]
        flags_ynpq = [False if i in ok_indices_ynpq else True for i in range(num_series)]

    return pd.DataFrame(flags_y2, index=group.index, columns=['flag_shape_y2']), pd.DataFrame(flags_ynpq, index=group.index, columns=['flag_shape_ynpq'])

def apply_shape_flagging_mutants(data, threshold_distances_y2, threshold_distances_ynpq, p=(2/3), threshold_variance = 1):
    data_copy = data.copy()
    data_copy['flag_shape_y2'] = np.nan
    data_copy['flag_shape_ynpq'] = np.nan
    # Apply the flagging function to each group and concatenate the results
    flagged_series_y2 = pd.concat([flag_time_series_shape_mutant(group, name, threshold_distances_y2, threshold_distances_ynpq, p, threshold_variance)[0] for name, group in data_copy.groupby(['mutated_genes', 'light_regime'])])
    flagged_series_ynpq = pd.concat([flag_time_series_shape_mutant(group, name, threshold_distances_y2, threshold_distances_ynpq, p, threshold_variance)[1] for name, group in data_copy.groupby(['mutated_genes', 'light_regime'])])

    # Merge the two flagged series back to the original DataFrame
    data_copy = data_copy.merge(flagged_series_y2, left_index=True, right_index=True, how='left')
    data_copy = data_copy.merge(flagged_series_ynpq, left_index=True, right_index=True, how='left')

    return data_copy

def get_shape_threshold_distances(data_smoothed, y):
    threshold_distances = {}
    for light in data_smoothed['light_regime'].unique():
        time_series = data_smoothed[(data_smoothed['mutant_ID'] == 'WT') & (data_smoothed['flag_' + y] == 'ok') & (data_smoothed['light_regime'] == light)].filter(like=y+'_').dropna(axis=1).values
        time_series_grad = np.gradient(time_series, axis=1)
        distance_matrix = dtw.distance_matrix(time_series_grad)
        mean_distance = np.mean(distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)])
        std_distance = np.std(distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)])
        threshold_distances[light] = mean_distance + 2*std_distance
    return threshold_distances

