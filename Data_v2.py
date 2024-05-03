import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Genes_self_similarity import get_mean_intra_distance_for_genes, get_mean_var_WT, apply_flagging, apply_flagging_2
from Model_checks import simulate_gaussian_vectors

def get_format_data() :
    data = pd.read_parquet('database_4-24-24.parquet')
    # data = pd.read_parquet('database_update.parquet')

    # replace the None in 'mutated_genes' by ''
    data['mutated_genes'] = data['mutated_genes'].replace({None: ''})

    # replace the rows that have 'mutated_genes' == '' and 'mutant_ID' != 'WT' by 'mutated_genes' == 'special_mutant'
    data.loc[(data['mutated_genes'] == '') & (data['mutant_ID'] != 'WT'), 'mutated_genes'] = 'special_mutant'

    data['mutated_genes_light_regime_count'] = data.groupby(['mutated_genes', 'light_regime'])['mutant_ID'].transform('count')

    # setting the last y_2 value appart as fv_fm_end

    # Create a new column 'fv_fm_end' and initialize it with NaN
    data['fv_fm_end'] = None
    data['ynpqend'] = None

    # Move 'y_81' to 'fv_fm_end' column for rows where 'num_frames' is 164
    data.loc[data['num_frames'] == 164, 'fv_fm_end'] = data.loc[data['num_frames'] == 164, 'y2_81']
    data.loc[data['num_frames'] == 164, 'ynpqend'] = data.loc[data['num_frames'] == 164, 'ynpq_81']

    # Replace 'y_41' with NaN for rows where 'num_frames' is 84
    data.loc[data['num_frames'] == 84, 'fv_fm_end'] = data.loc[data['num_frames'] == 84, 'y2_41']
    data.loc[data['num_frames'] == 84, 'y2_41'] = None

    data.loc[data['num_frames'] == 84, 'ynpqend'] = data.loc[data['num_frames'] == 84, 'ynpq_41']
    data.loc[data['num_frames'] == 84, 'ynpq_41'] = None

    # Drop 'y_81' column
    data.drop(columns=['y2_81'], inplace=True)
    data.drop(columns=['ynpq_81'], inplace=True)

    # Convert well_id to numerical representation
    data['well_id_numeric'] = pd.factorize(data['well_id'])[0]

    # Calculate mean of y2_ columns
    data['mean_y2'] = data.filter(like='y2_').mean(axis=1)
    data['mean_ynpq'] = data.filter(like='ynpq_').mean(axis=1)

    data['median'] = None
    data['log_likelihood_null'] = None

    return data

def get_format_data_without_na() :
    data = get_format_data()

    # Remove the rows that have an anomaly in their number of frames or fv_fm values
    data = data[data['num_frames'] <= 164]
    data = data.dropna(subset=['fv_fm'])

    # Remove the rows of data['plate'] == 15 and data['mutant_ID'] == 'WT' and data['well_id'] == 'N03'
    data = data[~((data['plate'] == 15) & (data['mutant_ID'] == 'WT') & (data['well_id'] == 'N03'))]

    return data.reset_index(drop=True)

def get_split_genes(data):
    # Create a new DataFrame to store the expanded rows
    expanded_rows = []

    # Iterate over each row in the original DataFrame
    for index, row in data.iterrows():
        genes = row['mutated_genes']
        # Split genes based on comma (,) or ampersand (&)
        gene_list = [gene.strip() for gene in genes.replace('&', ',').split(',')]
        
        # Create a new row for each gene
        for gene in gene_list:
            new_row = row.copy()  # Create a copy of the original row
            new_row['mutated_genes'] = gene  # Replace mutated_genes with the current gene
            expanded_rows.append(new_row)  # Append the new row to the expanded_rows list

    # Create a new DataFrame from the expanded_rows list
    expanded_df = pd.DataFrame(expanded_rows)

    # Reset index of the new DataFrame
    expanded_df.reset_index(drop=True, inplace=True)

    return expanded_df

def get_normalize_data() :
    data = get_format_data_without_na()
    data_WT = data[data['mutant_ID'] == 'WT']

    data_normalized = data.copy()

    target_fv_fm = data_WT['fv_fm'].mean()

    # Calculate the average fv_fm for the WT mutants for each combination of 'plate' and 'light_regime'
    average_fv_fm_WT = data_normalized[data_normalized['mutant_ID'] == 'WT'].groupby(['plate', 'light_regime'])['fv_fm'].mean().reset_index()

    # Merge the average fv_fm values back into the main DataFrame
    data_normalized = data_normalized.merge(average_fv_fm_WT, on=['plate', 'light_regime'], suffixes=('', '_WT'))

    # Calculate the ratio of the average fv_fm values and the target fv_fm value
    data_normalized['fv_fm_ratio'] = data_normalized['fv_fm_WT'] / target_fv_fm

    # Divide y2 values by the ratio of the average fv_fm values
    for col in data_normalized.filter(like='y2_').columns:
        data_normalized[col] /= data_normalized['fv_fm_ratio']

    data_normalized['fv_fm'] = data_normalized['fv_fm']/data_normalized['fv_fm_ratio']

    data_ncopy = data_normalized.copy()

    grouped = data_ncopy.groupby(['plate', 'light_regime', 'mutant_ID'])

    wt_means = {}

    for name, group in grouped:
        if name[2] == 'WT':
            wt_means[name] = group.filter(like='y2_').mean()
    
    # Iterate over each row in DataFrame
    for index, row in data_ncopy.iterrows():
        plate = row['plate']
        light_regime = row['light_regime']
        
        # Retrieve the associated time-series from the dictionary
        wt_mean = wt_means.get((plate, light_regime, 'WT'))
        
        if wt_mean is not None:
            # Subtract the time-series from the corresponding 'y2_i' columns
            for i in range(1, 81):  # Assuming you have 'y2_1' to 'y2_80' columns
                column_name = f'y2_{i}'
                if data_ncopy.at[index, column_name] is None:
                    continue
                data_ncopy.at[index, column_name] -= wt_mean[i - 1]
    
    return data_ncopy.reset_index(drop=True)

def normalize_data_additive(data) :
    data_copy = data.copy()
    # Calculate the average fv_fm for the WT mutants for each combination of 'plate' and 'light_regime'
    average_fv_fm_WT = data_copy[data_copy['mutant_ID'] == 'WT'].groupby(['plate', 'light_regime'])['fv_fm'].mean().reset_index()
    average_fv_fm_end_WT = data_copy[data_copy['mutant_ID'] == 'WT'].groupby(['plate', 'light_regime'])['fv_fm_end'].mean().reset_index()
    average_end_ynpq_WT = data_copy[data_copy['mutant_ID'] == 'WT'].groupby(['plate', 'light_regime'])['ynpqend'].mean().reset_index()

    # Merge the average fv_fm values back into the main DataFrame
    data_copy = data_copy.merge(average_fv_fm_WT, on=['plate', 'light_regime'], suffixes=('', '_WT'))
    data_copy = data_copy.merge(average_fv_fm_end_WT, on=['plate', 'light_regime'], suffixes=('', '_WT'))
    data_copy = data_copy.merge(average_end_ynpq_WT, on=['plate', 'light_regime'], suffixes=('', '_WT'))

    grouped = data_copy.groupby(['plate', 'light_regime', 'mutant_ID'])

    wt_means_y2 = {}
    wt_means_ynpq = {}

    for name, group in grouped:
        if name[2] == 'WT':
            if 'ok' not in group['flag_y2'].values :
                wt_means_y2[name] = group.filter(like='y2_').mean()
            if 'ok' in group['flag_y2'].values :
                wt_means_y2[name] = group[group['flag_y2'] == 'ok'].filter(like='y2_').mean()
            if 'ok' not in group['flag_ynpq'].values :
                wt_means_ynpq[name] = group.filter(like='ynpq_').mean()
            if 'ok' in group['flag_ynpq'].values :
                wt_means_ynpq[name] = group[group['flag_ynpq'] == 'ok'].filter(like='ynpq_').mean()
    
    # Iterate over each row in DataFrame
    for index, row in data_copy.iterrows():
        plate = row['plate']
        light_regime = row['light_regime']
        
        # Retrieve the associated time-series from the dictionary
        wt_mean_y2 = wt_means_y2.get((plate, light_regime, 'WT'))
        wt_mean_ynpq = wt_means_ynpq.get((plate, light_regime, 'WT'))
        
        if wt_mean_y2 is not None:
            # Subtract the time-series from the corresponding 'y2_i' columns
            for i in range(1, 81):  # Assuming you have 'y2_1' to 'y2_80' columns
                column_name = f'y2_{i}'
                if data_copy.at[index, column_name] is None:
                    continue
                data_copy.at[index, column_name] -= wt_mean_y2[i - 1]

        if wt_mean_ynpq is not None:
            # Subtract the time-series from the corresponding 'y2_i' columns
            for i in range(1, 81):  # Assuming you have 'y2_1' to 'y2_80' columns
                column_name = f'ynpq_{i}'
                if data_copy.at[index, column_name] is None:
                    continue
                data_copy.at[index, column_name] -= wt_mean_ynpq[i - 1]

    data_copy['fv_fm'] = data_copy['fv_fm'] - data_copy['fv_fm_WT']
    data_copy['fv_fm_end'] = data_copy['fv_fm_end'] - data_copy['fv_fm_end_WT']

    data_copy['ynpqend'] = data_copy['ynpqend'] - data_copy['ynpqend_WT']

    data_copy['mean_y2'] = data_copy.filter(like='y2_').mean(axis=1)
    data_copy['mean_ynpq'] = data_copy.filter(like='ynpq_').mean(axis=1)
    
    return data_copy

def normalize_data_additive_2(data) :
    data_copy = data.copy()
    grouped = data_copy.groupby(['plate', 'light_regime', 'mutant_ID'])

    wt_means = {}

    for name, group in grouped:
        if name[2] == 'WT':
            wt_means[name] = group.filter(like='y2_').mean()
    
    # Iterate over each row in DataFrame
    for index, row in data_copy.iterrows():
        plate = row['plate']
        light_regime = row['light_regime']
        
        # Retrieve the associated time-series from the dictionary
        wt_mean = wt_means.get((plate, light_regime, 'WT'))
        
        if wt_mean is not None:
            # Subtract the time-series from the corresponding 'y2_i' columns
            for i in range(1, 81):  # Assuming you have 'y2_1' to 'y2_80' columns
                column_name = f'y2_{i}'
                if data_copy.at[index, column_name] is None:
                    continue
                data_copy.at[index, column_name] -= wt_mean[i - 1]
    
    # data_copy['fv_fm'] = data_copy['fv_fm'] - data_copy['fv_fm_WT']

    data_copy['mean_y2'] = data_copy.filter(like='y2_').mean(axis=1)
    
    return data_copy

def get_data_norm_flagged(data_norm, p):
    intra_distance_norm_df = get_mean_intra_distance_for_genes(data_norm)
    intra_gene_distance_WT_norm_y2, intra_gene_distance_WT_norm_ynpq = get_mean_var_WT(intra_distance_norm_df, type='mean_intra_gene_distance')
    data_norm_flagged = apply_flagging(data_norm, threshold_distance_y2=2*intra_gene_distance_WT_norm_y2, threshold_distance_ynpq=2*intra_gene_distance_WT_norm_ynpq, p=p, threshold_variance=1)
    return data_norm_flagged.reset_index(drop=True)

def get_data_norm_flagged_2(data_norm):
    data_norm_flagged = apply_flagging_2(data_norm, threshold_variance=1)
    return data_norm_flagged.reset_index(drop=True)

def get_gene_data(data) :
    data_norm_flagged = get_data_norm_flagged(data)
    data_norm_ok = data_norm_flagged[data_norm_flagged['flag'] == 'ok']
    group_cols = ['light_regime', 'mutated_genes']
    y2_cols = [col for col in data_norm_ok.columns if col.startswith('y2_')]
    aggregations = {'fv_fm': 'mean'}  
    aggregations.update({'fv_fm_end': 'mean'})
    aggregations.update({col: 'mean' for col in y2_cols}) 
    aggregations.update({'mean_y2': 'mean'})
    aggregations.update({'num_frames': 'mean'})
    # Group by 'light_regime' and 'mutated_genes', calculate mean of 'fv_fm' and 'y2' columns
    data_gene = data_norm_ok.groupby(group_cols, as_index=False).agg(aggregations)
    data_gene['log_likelihood_null'] = None
    return data_gene

def get_rolling_average_data(data) :
    # Create a list to store the columns after replacement
    data_rolling = data.copy()
    y2_cols = [col for col in data.columns if col.startswith("y2_")]
    data_rolling[y2_cols] = data[y2_cols].T.rolling(window=5, min_periods=1).mean().T
    # when num_frames = 84, set y2_41, y2_42, y2_43, y2_44, y2_45 to NaN
    data_rolling.loc[data_rolling['num_frames'] == 84, ['y2_41', 'y2_42', 'y2_43', 'y2_44', 'y2_45']] = np.nan
    return data_rolling

    # new_columns = []
    # window_size = 5 

    # # Iterate over the columns of the DataFrame
    # for col in data.columns:
    #     if col.startswith('y2_') and col[3:].isdigit():  # Check if the column starts with 'y2_' followed by digits
    #         # Calculate the rolling average for the column
    #         rolling_avg = data[col].rolling(window=window_size).mean()
    #         # Store the rolling average as a new column
    #         new_columns.append(rolling_avg)
    #     else:
    #         # Store columns that don't match the format 'y2_i' unchanged
    #         new_columns.append(data[col])

    # # Create a new DataFrame with the modified columns
    # return pd.concat(new_columns, axis=1)

def get_good_outliers(data_norm_ok) :
    list_mutant_good = data_norm_ok['mutant_ID'].value_counts()[data_norm_ok['mutant_ID'].value_counts() >= 5]
    outliers_all = data_norm_ok[(data_norm_ok['outlier_euclidian_distance'] == True) & (data_norm_ok['outlier_median_distance'] == True) & (data_norm_ok['outlier_mean_norm_std'] == True)]
    outliers_all_good = outliers_all[outliers_all['mutant_ID'].isin(list_mutant_good.index)]
    genes_lights = {}
    for light in data_norm_ok['light_regime'].unique():
        outliers_mutant_good_light = outliers_all_good[outliers_all_good['light_regime'] == light]
        multiple_mutants_genes_good_light = outliers_mutant_good_light['mutated_genes'].value_counts()
        genes_lights[light] = multiple_mutants_genes_good_light
    df = pd.DataFrame(genes_lights).T.fillna(0)

    # keep the columns where at least one row is ge 2
    df = df[df.columns[df.gt(1).any()]]
    return df

def get_train_test_WT(data_norm_ok) :
    data_test = data_norm_ok[data_norm_ok['plate'] <= 3]
    # get 10% of the data of plate 99
    data_test_99 = data_norm_ok[data_norm_ok['plate'] == 99].sample(frac=0.1, random_state=42)
    data_test = pd.concat([data_test, data_test_99])
    data_train = data_norm_ok[~data_norm_ok.index.isin(data_test.index)]
    return data_train, data_test

def replace_WT_by_model(data_norm_ok, cov_matrices) :
    # replace each row of data_norm_ok where mutant_ID == 'WT' by simulate_gaussian_vector(n, d, cov)
    data_norm_ok_copy = data_norm_ok.copy()
    for index, row in data_norm_ok_copy.iterrows():
        if row['mutant_ID'] == 'WT':
            n = 1
            d = (row['num_frames']//2 - 2)
            data_norm_ok_copy.loc[index, 'y2_1':'y2_' + str(d)] = simulate_gaussian_vectors(n, np.zeros(d), cov_matrices[row['light_regime']])[0]
    return data_norm_ok_copy

def replace_WT_and_M_by_model(data_norm_ok, cov_matrices) :
    # replace each row of data_norm_ok where mutant_ID == 'WT' by simulate_gaussian_vector(n, d, cov)
    data_norm_ok_copy = data_norm_ok.copy()
    for index, row in data_norm_ok_copy.iterrows():
        if row['mutant_ID'] == 'WT' and row['plate'] == 99:
            n = 1
            d = (row['num_frames']//2 - 2)
            data_norm_ok_copy.loc[index, 'y2_1':'y2_' + str(d)] = simulate_gaussian_vectors(n, np.zeros(d), (2/3)*cov_matrices[row['light_regime']])[0]
        elif row['mutant_ID'] == 'WT' and row['plate'] != 99:
            n = 1
            d = (row['num_frames']//2 - 2)
            data_norm_ok_copy.loc[index, 'y2_1':'y2_' + str(d)] = simulate_gaussian_vectors(n, np.zeros(d), (1 - 1/384)*cov_matrices[row['light_regime']])[0]
        else :
            n = 1
            d = (row['num_frames']//2 - 2)
            # simulate a 1/2 Bernouilli
            b = np.random.randint(0, 2)
            # create a random that is not zero if b = 1
            if b == 1:
                mean = np.random.uniform(0.02, 0.15, d)
                c = np.random.randint(0, 2)
                if c == 1:
                    mean = -mean
                data_norm_ok_copy.loc[index, 'y2_1':'y2_' + str(d)] = simulate_gaussian_vectors(n, mean, (4/3)*cov_matrices[row['light_regime']])[0]
            else:
                data_norm_ok_copy.loc[index, 'y2_1':'y2_' + str(d)] = simulate_gaussian_vectors(n, np.zeros(d), (4/3)*cov_matrices[row['light_regime']])[0]
    return data_norm_ok_copy
