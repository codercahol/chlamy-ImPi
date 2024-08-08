import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import Genes_self_similarity_v2
from Model_checks import simulate_gaussian_vectors
from scipy.ndimage import gaussian_filter1d
from Feature_engineering import kernel_smooth, local_smooth
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

def check_database_errors():
    errors = []
    
    # Check if the Parquet file exists and can be read
    # parquet_files = ['database_4-24-24.parquet']
    parquet_files = ['20240721_database.parquet']
    for file in parquet_files:
        if not os.path.exists(file):
            errors.append(f"File {file} does not exist.")
        else:
            try:
                data = pd.read_parquet(file)
                break
            except Exception as e:
                errors.append(f"Failed to read {file}: {str(e)}")
    else:
        return errors  # If none of the parquet files could be read, return the errors

    # Check if the necessary columns are present in the data
    required_columns = ['mutated_genes', 'feature', 'mutant_ID', 'num_frames', 'y2_81', 'ynpq_81', 'y2_41', 'ynpq_41', 'well_id', 'plate', 'light_regime', 'fv_fm']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing columns in data: {', '.join(missing_columns)}")

    # Check if y2_81 and ynpq_81 columns are the last columns of y2
    y2_columns = data.filter(like='y2_').columns
    if not y2_columns[-1] == 'y2_81':
        errors.append("y2_81 is not the last column in the DataFrame.")
    
    # Check for 'num_frames' anomalies
    if not all(data['num_frames'].isin([164, 84])):
        errors.append("There are unexpected values in 'num_frames' column.")

    # Check if the gene information file exists and can be read
    gene_info_file = 'CreinhardtiiCC_4532_707_v6.1.annotation_info.txt'
    if not os.path.exists(gene_info_file):
        errors.append(f"File {gene_info_file} does not exist.")
    else:
        try:
            gene_info = pd.read_csv(gene_info_file, sep='\t')
        except Exception as e:
            errors.append(f"Failed to read {gene_info_file}: {str(e)}")
    
    # Check necessary columns in gene_info
    if gene_info is not None:
        required_gene_info_columns = ['Best-hit-clamy-name', 'GO']
        missing_gene_info_columns = [col for col in required_gene_info_columns if col not in gene_info.columns]
        if missing_gene_info_columns:
            errors.append(f"Missing columns in gene_info: {', '.join(missing_gene_info_columns)}")

    if errors:
        return errors
    else:
        return "No errors found."

def get_format_data() :
    # data = pd.read_parquet('database_4-24-24.parquet')
    # data = pd.read_parquet('/Users/samsongourevitch/Documents/Chlamy_Project_Github/Chlamy_project/database_4-24-24.parquet')
    data = pd.read_parquet('/Users/samsongourevitch/Documents/Chlamy_Project_Github/Chlamy_project/20240721_database.parquet')
    # data = pd.read_parquet('20240601_database.parquet')
    # data = pd.read_parquet('database_update.parquet')

    # replace the None in 'mutated_genes' by ''
    data['mutated_genes'] = data['mutated_genes'].replace({None: ''})
    data['feature'] = data['feature'].replace({None: ''})

    # replace the rows that have 'mutated_genes' == '' and 'mutant_ID' != 'WT' by 'mutated_genes' == 'special_mutant'
    data.loc[(data['mutated_genes'] == '') & (data['mutant_ID'] != 'WT'), 'mutated_genes'] = 'special_mutant'

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

    # # Drop 'y_81' column
    # data.drop(columns=['y2_81'], inplace=True)
    # data.drop(columns=['ynpq_81'], inplace=True)

    data['new_record_fv_fm'] = True

    # Convert measurement time columns to datetime
    time_columns = [col for col in data.columns if 'measurement_time' in col]
    for col in time_columns:
        data[col] = pd.to_datetime(data[col])

    # for the rows where (data[time_columns[1]] - data[time_columns[0]]).dt.total_seconds() / 3600 > 1, replace data[time_columns[0]] by data[time_columns[1]] - 30min and set new_record_fv_fm to False
    data.loc[(data[time_columns[1]] - data[time_columns[0]]).dt.total_seconds() / 3600 > 10, 'new_record_fv_fm'] = False
    data.loc[(data[time_columns[1]] - data[time_columns[0]]).dt.total_seconds() / 3600 > 10, time_columns[0]] = data[time_columns[1]] - pd.Timedelta('30 min')

    # Calculate elapsed time in hours from the first measurement
    for i, col in enumerate(time_columns):
        if i == 0:
            data['elapsed_time_0'] = 0
        else:
            data[f'elapsed_time_{i}'] = (data[col] - data[time_columns[0]]).dt.total_seconds() / 3600

    # Convert well_id to numerical representation
    data['well_id_numeric'] = pd.factorize(data['well_id'])[0]

    # Calculate mean of y2_ columns
    data['mean_y2'] = data.filter(like='y2_').mean(axis=1)
    data['mean_ynpq'] = data.filter(like='ynpq_').mean(axis=1)

    data['median'] = None
    data['log_likelihood_null'] = None

    # gene_info = pd.read_csv('CreinhardtiiCC_4532_707_v6.1.annotation_info.txt', sep='\t')
    gene_info = pd.read_csv('/Users/samsongourevitch/Documents/Chlamy_Project_Github/Chlamy_project/CreinhardtiiCC_4532_707_v6.1.annotation_info.txt', sep='\t')
    gene_GO = gene_info[['Best-hit-clamy-name', 'GO']].dropna()
    gene_GO.rename(columns={'Best-hit-clamy-name': 'mutated_genes'}, inplace=True)
    gene_GO['GO'] = gene_GO['GO'].apply(lambda x: x.split(' '))
    gene_GO.drop_duplicates(subset='mutated_genes', inplace=True)

    data = pd.merge(data, gene_GO, on='mutated_genes', how='left')
    data['GO'] = data['GO'].apply(lambda x: [] if not isinstance(x, list) and pd.isna(x) else x)

    return data

def get_format_data_new_database() :
    # data = pd.read_parquet('database_4-24-24.parquet')
    # data = pd.read_parquet('/Users/samsongourevitch/Documents/Chlamy_Project_Github/Chlamy_project/database_4-24-24.parquet')
    data = pd.read_parquet('/Users/samsongourevitch/Documents/Chlamy_Project_Github/Chlamy_project/20240721_database.parquet')
    # data = pd.read_parquet('20240601_database.parquet')
    # data = pd.read_parquet('database_update.parquet')

    # replace the None in 'mutated_genes' by ''
    data['mutated_genes'] = data['mutated_genes'].replace({None: ''})
    data['feature'] = data['feature'].replace({None: ''})

    # replace the rows that have 'mutated_genes' == '' and 'mutant_ID' != 'WT' by 'mutated_genes' == 'special_mutant'
    data.loc[(data['mutated_genes'] == '') & (data['mutant_ID'] != 'WT'), 'mutated_genes'] = 'special_mutant'

    # setting the last y_2 value appart as fv_fm_end

    # Create a new column 'fv_fm_end' and initialize it with NaN
    data['fv_fm_end'] = None
    data['ynpqend'] = None

    y2_columns = [col for col in data.columns if col.startswith('y2_')]
    ynpq_columns = [col for col in data.columns if col.startswith('ynpq_')]

    # Create a new column fv_fm_end for the last y2 value
    data['fv_fm_end'] = data[y2_columns].apply(lambda row: row.dropna().iloc[-1] if not row.dropna().empty else np.nan, axis=1)
    data['ynpqend'] = data[ynpq_columns].apply(lambda row: row.dropna().iloc[-1] if not row.dropna().empty else np.nan, axis=1)

    # Replace the last y2 value with NaN
    for index, row in data.iterrows():
        last_y2_col = row.filter(like='y2_').last_valid_index()
        last_ynpq_col = row.filter(like='ynpq_').last_valid_index()
        data.at[index, last_y2_col] = np.nan
        data.at[index, last_ynpq_col] = np.nan

    # # Drop 'y_81' column
    # data.drop(columns=['y2_81'], inplace=True)
    # data.drop(columns=['ynpq_81'], inplace=True)

    data = data.dropna(axis=1, how='all')

    data['new_record_fv_fm'] = True

    # Convert measurement time columns to datetime
    time_columns = [col for col in data.columns if 'measurement_time' in col]
    for col in time_columns:
        data[col] = pd.to_datetime(data[col])

    # for the rows where (data[time_columns[1]] - data[time_columns[0]]).dt.total_seconds() / 3600 > 1, replace data[time_columns[0]] by data[time_columns[1]] - 30min and set new_record_fv_fm to False
    data.loc[(data[time_columns[1]] - data[time_columns[0]]).dt.total_seconds() / 3600 > 10, 'new_record_fv_fm'] = False
    data.loc[(data[time_columns[1]] - data[time_columns[0]]).dt.total_seconds() / 3600 > 10, time_columns[0]] = data[time_columns[1]] - pd.Timedelta('30 min')

    # Calculate elapsed time in hours from the first measurement
    for i, col in enumerate(time_columns):
        if i == 0:
            data['elapsed_time_0'] = 0
        else:
            data[f'elapsed_time_{i}'] = (data[col] - data[time_columns[0]]).dt.total_seconds() / 3600

    # Convert well_id to numerical representation
    data['well_id_numeric'] = pd.factorize(data['well_id'])[0]

    # Calculate mean of y2_ columns
    data['mean_y2'] = data.filter(like='y2_').mean(axis=1)
    data['mean_ynpq'] = data.filter(like='ynpq_').mean(axis=1)

    # gene_info = pd.read_csv('CreinhardtiiCC_4532_707_v6.1.annotation_info.txt', sep='\t')
    gene_info = pd.read_csv('/Users/samsongourevitch/Documents/Chlamy_Project_Github/Chlamy_project/CreinhardtiiCC_4532_707_v6.1.annotation_info.txt', sep='\t')
    gene_GO = gene_info[['Best-hit-clamy-name', 'GO']].dropna()
    gene_GO.rename(columns={'Best-hit-clamy-name': 'mutated_genes'}, inplace=True)
    gene_GO['GO'] = gene_GO['GO'].apply(lambda x: x.split(' '))
    gene_GO.drop_duplicates(subset='mutated_genes', inplace=True)

    data = pd.merge(data, gene_GO, on='mutated_genes', how='left')
    data['GO'] = data['GO'].apply(lambda x: [] if not isinstance(x, list) and pd.isna(x) else x)

    return data

def split_time_series(row, regime):
    high_light = []
    low_light = []
    if regime == '2h-2h':
        for i in range(0, len(row), 8):
            high_light.extend(row[i:i+4])
            low_light.extend(row[i+4:i+8])
    elif regime == '10min-10min':
        low_light.append(row[0])
        for i in range(1, len(row), 4):
            high_light.extend(row[i:i+2])
            low_light.extend(row[i+2:i+4])
    elif regime in ['1min-1min', '30s-30s']:
        for i in range(0, len(row), 2):
            high_light.append(row[i])
            low_light.append(row[i+1])
    return high_light, low_light

def get_split_time_series(data):
    # Define the columns containing the time series data
    time_series_columns_y2 = [col for col in data.columns if col.startswith('y2_')]
    time_series_columns_ynpq = [col for col in data.columns if col.startswith('ynpq_')]

    measured_time_columns = [col for col in data.columns if 'measurement_time' in col]
    elapsed_time_columns = [col for col in data.columns if 'elapsed_time' in col]

    # Create new DataFrame to store the results
    result_rows = []

    # Apply the splitting function
    for idx, row in data.iterrows():
        if row['light_regime'] in ['2h-2h', '10min-10min', '1min-1min', '30s-30s']:
            high_light_y2, low_light_y2 = split_time_series(row[time_series_columns_y2].values, row['light_regime'])
            high_light_ynpq, low_light_ynpq = split_time_series(row[time_series_columns_ynpq].values, row['light_regime'])

            high_light_measure_time, low_light_measure_time = split_time_series(row[measured_time_columns].dropna().values[1:-1], row['light_regime'])
            high_light_elapsed_time, low_light_elapsed_time = split_time_series(row[elapsed_time_columns].dropna().values[1:-1], row['light_regime'])

            high_row = row.copy()
            high_row[time_series_columns_y2[:len(high_light_y2)]] = high_light_y2
            high_row[time_series_columns_y2[len(high_light_y2):]] = np.nan 
            high_row[time_series_columns_ynpq[:len(high_light_ynpq)]] = high_light_ynpq
            high_row[time_series_columns_ynpq[len(high_light_ynpq):]] = np.nan
            high_row[measured_time_columns[0]] = row[measured_time_columns].values[0]
            high_row[measured_time_columns[1:len(high_light_measure_time) + 1]] = high_light_measure_time
            high_row[measured_time_columns[len(high_light_measure_time)+1:]] = np.nan
            high_row[measured_time_columns[0]] = row[elapsed_time_columns].values[0]
            high_row[elapsed_time_columns[1:len(high_light_elapsed_time) + 1]] = high_light_elapsed_time
            high_row[elapsed_time_columns[len(high_light_elapsed_time)+1:]] = np.nan
            high_row['light_regime'] = 'high_' + row['light_regime']
            # Create low light row
            low_row = row.copy()
            low_row[time_series_columns_y2[:len(low_light_y2)]] = low_light_y2
            low_row[time_series_columns_y2[len(low_light_y2):]] = np.nan
            low_row[time_series_columns_ynpq[:len(low_light_ynpq)]] = low_light_ynpq
            low_row[time_series_columns_ynpq[len(low_light_ynpq):]] = np.nan
            low_row[measured_time_columns[:len(low_light_measure_time)]] = low_light_measure_time
            low_row[measured_time_columns[len(low_light_measure_time)]] = row[measured_time_columns].dropna().values[-1]
            low_row[measured_time_columns[len(low_light_measure_time)+1:]] = np.nan
            low_row[elapsed_time_columns[:len(low_light_elapsed_time)]] = low_light_elapsed_time
            low_row[elapsed_time_columns[len(low_light_elapsed_time)]] = row[elapsed_time_columns].dropna().values[-1]
            low_row[elapsed_time_columns[len(low_light_elapsed_time)+1:]] = np.nan
            low_row['light_regime'] = 'low_' + row['light_regime']
            # Append to result DataFrame
            result_rows.append(high_row)
            result_rows.append(low_row)
        else:
            result_rows.append(row)

    result = pd.DataFrame(result_rows)
    result['mean_y2'] = result.filter(like='y2_').mean(axis=1)
    result['mean_ynpq'] = result.filter(like='ynpq_').mean(axis=1)
    
    return result


def get_format_data_without_na(phase='phase1') :
    data = get_format_data_new_database()
    # rename mutant_ID WTF by WT
    data['mutant_ID'] = data['mutant_ID'].replace({'WTF': 'WT'})
    if phase == 'phase1':
        data = data[(data['start_date'] < '2024-05-19')]   
    elif phase == 'transition':
        data = data[((data['plate'] == '20') | (data['plate'] == '99')) & (data['start_date'] >= '2024-05-19')]
    elif phase == 'phase2':
        data = data[(data['plate'] != '20') & (data['plate'] != '99') & (data['start_date'] >= '2024-05-19')]
        
    data = get_split_genes_and_features(data)

    # Remove the rows that have an anomaly in their number of frames or fv_fm values
    # data = data[data['num_frames'] <= 164]
    data = data.dropna(subset=['fv_fm'])

    # Remove the rows of data['plate'] == 15 and data['mutant_ID'] == 'WT' and data['well_id'] == 'N03'
    data = data[~((data['plate'] == 15) & (data['mutant_ID'] == 'WT') & (data['well_id'] == 'N03'))]

    data['mutated_genes_light_regime_count'] = data.groupby(['mutated_genes', 'light_regime'])['mutant_ID'].transform('count')
    
    return data.reset_index(drop=True)

def get_split_genes_and_features(data):
    # Create a new DataFrame to store the expanded rows
    expanded_rows = []

    # Iterate over each row in the original DataFrame
    for index, row in data.iterrows():
        genes = row['mutated_genes']
        features = row['feature']

        # Split genes and features based on comma (,) or ampersand (&)
        gene_list = [gene.strip() for gene in genes.replace('&', ',').split(',')]
        feature_list = [feature.strip() for feature in features.replace('&', ',').split(',')]

        # Ensure the lengths of gene_list and feature_list match
        if len(gene_list) != len(feature_list):
            # raise ValueError(f"Row {index} has mismatched gene and feature counts: {genes} vs {features}")
            for gene in gene_list:
                new_row = row.copy() 
                new_row['mutated_genes'] = gene 
                expanded_rows.append(new_row)
        else :
            # Create a new row for each gene-feature pair
            for gene, feature in zip(gene_list, feature_list):
                new_row = row.copy()  # Create a copy of the original row
                new_row['mutated_genes'] = gene  # Replace mutated_genes with the current gene
                new_row['feature'] = feature  # Replace feature with the current feature
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

    background_WT = {'CC125': 'CC125', 'pgrl1-86-8' : 'CC125', 'pgr5C1': 'CC125', 'pgrl1-pgr5-38' : 'CC125', 'stt7-11-2' : 'CC125', 'pgr5-97-2' : 'CC125', 'pgr5 2b32': 'CC125', 'aox 1-5' : 'CC125', 'aox 1-5 comp' : 'CC125',
                     '137AH': '137AH', 'pgrl1': '137AH', 'Cp2A6': '137AH',
                     'WT1': 'WT1', 'DM1': 'WT1', 
                     'WT2': 'WT2', 'DM2': 'WT2',
                     'WT3': 'WT3', 'DM3': 'WT3',
                     'WT4': 'WT4', 'DM4': 'WT4',
                     'WT5': 'WT5', 'DM5': 'WT5',
                     '4a+': '4a+', '4a-': '4a-',
                     'C137+': 'C137+',
                     'CC5133':  'CC5133', 
                     'nda3-91': 'CC125', 'nda2-91': 'CC125', 'nda2-75': 'CC125', 'nda2-12': 'CC125', 'nda3-119': 'CC125',
                     'ptox 2.24': 'CC125', 
                     'Cp2A5': '137AH', 
                     'D66': 'D66', 'mdh 2-3': 'D66',
                     'CC4533': 'CC4533',
                     'npq4 mt-': '4a-', 
                     'npq4lhcsr1': '4a+', 
                     'CC5155': 'CC5155', 
                     'cia5': 'cia5 compa', 'cia5 compb': 'cia5 compa', 'cia5 compa': 'cia5 compa', 
                     'CC5325-JF': 'CC5325-JF', 
                     'PSBS1/2N3:20': 'CC125', 'NNT1:79': 'CC125', 'NNT3:48': 'CC125', 'CC125/C131': 'CC125', 'mdh5-30': 'CC125', 'miro-66': 'CC125', 'mdh5-62': 'CC125',
                     'NK1NK2-15': '21gr+', 'NK2-7': '21gr+', 'lci2': '21gr+', 
                     'cah4/5-ccp1/2-282': 'CC125-JF'}

    # Calculate the average fv_fm for the WT mutants for each combination of 'plate' and 'light_regime'
    average_fv_fm_WT = data_copy[data_copy['mutant_ID'] == 'WT'].groupby(['plate', 'light_regime'])['fv_fm'].mean().reset_index()
    average_fv_fm_end_WT = data_copy[data_copy['mutant_ID'] == 'WT'].groupby(['plate', 'light_regime'])['fv_fm_end'].mean().reset_index()
    average_end_ynpq_WT = data_copy[data_copy['mutant_ID'] == 'WT'].groupby(['plate', 'light_regime'])['ynpqend'].mean().reset_index()

    # Merge the average fv_fm values back into the main DataFrame
    data_copy = data_copy.merge(average_fv_fm_WT, on=['plate', 'light_regime'], suffixes=('', '_WT'))
    data_copy = data_copy.merge(average_fv_fm_end_WT, on=['plate', 'light_regime'], suffixes=('', '_WT'))
    data_copy = data_copy.merge(average_end_ynpq_WT, on=['plate', 'light_regime'], suffixes=('', '_WT'))

    grouped = data_copy.groupby(['plate', 'light_regime', 'mutant_ID'])

    y2_cols = [col for col in data_copy.columns if col.startswith('y2_')]
    ynpq_cols = [col for col in data_copy.columns if col.startswith('ynpq_')]

    wt_means_y2 = {}
    wt_means_ynpq = {}

    special_wt_means_y2 = {}
    special_wt_means_ynpq = {}

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
        elif name[2] in background_WT.values():
            special_wt_means_y2[name] = group.filter(like='y2_').mean()
            special_wt_means_ynpq[name] = group.filter(like='ynpq_').mean()
    
    # Iterate over each row in DataFrame
    for index, row in data_copy.iterrows():
        plate = row['plate']
        light_regime = row['light_regime']

        if row['mutant_ID'] in background_WT.keys():
            # Retrieve the associated time-series from the dictionary
            wt_mean_y2 = special_wt_means_y2.get((plate, light_regime, background_WT[row['mutant_ID']]))
            wt_mean_ynpq = special_wt_means_ynpq.get((plate, light_regime, 'CC125'))

        else :
            # Retrieve the associated time-series from the dictionary
            wt_mean_y2 = wt_means_y2.get((plate, light_regime, 'WT'))
            wt_mean_ynpq = wt_means_ynpq.get((plate, light_regime, 'WT'))
        
        if wt_mean_y2 is not None:
            # Subtract the time-series from the corresponding 'y2_i' columns
            for i, col in enumerate(y2_cols):
                if data_copy.at[index, col] is None:
                    continue
                data_copy.at[index, col] -= wt_mean_y2[i]

        if wt_mean_ynpq is not None:
            # Subtract the time-series from the corresponding 'y2_i' columns
            for i, col in enumerate(ynpq_cols):
                if data_copy.at[index, col] is None:
                    continue
                data_copy.at[index, col] -= wt_mean_ynpq[i]

    data_copy['fv_fm'] = data_copy['fv_fm'] - data_copy['fv_fm_WT']
    data_copy['fv_fm_end'] = data_copy['fv_fm_end'] - data_copy['fv_fm_end_WT']

    data_copy['ynpqend'] = data_copy['ynpqend'] - data_copy['ynpqend_WT']

    data_copy['mean_y2'] = data_copy.filter(like='y2_').mean(axis=1)
    data_copy['mean_ynpq'] = data_copy.filter(like='ynpq_').mean(axis=1)
    
    return data_copy

def get_data_norm_flagged(data_norm, p):
    intra_distance_norm_df = get_mean_intra_distance_for_genes(data_norm)
    intra_gene_distance_WT_norm_y2, intra_gene_distance_WT_norm_ynpq = get_mean_var_WT(intra_distance_norm_df, type='mean_intra_gene_distance')
    data_norm_flagged = apply_flagging(data_norm, threshold_distance_y2=2*intra_gene_distance_WT_norm_y2, threshold_distance_ynpq=2*intra_gene_distance_WT_norm_ynpq, p=p, threshold_variance=1)
    return data_norm_flagged.reset_index(drop=True)

def get_data_norm_flagged_2(data_norm):
    data_norm_flagged = apply_flagging_2(data_norm, threshold_variance=1)
    return data_norm_flagged.reset_index(drop=True)

def concat_lists(series):
    return sum(series, [])

def get_gene_data_y2(data_slopes):
    # data_norm_flagged = get_data_norm_flagged(data)
    # data_norm_ok = data_norm_flagged[data_norm_flagged['flag_y2'] == 'ok']
    # Define the columns for grouping and aggregation
    group_cols = ['light_regime', 'mutated_genes']
    y2_cols = [col for col in data_slopes.columns if col.startswith('y2_')]

    # Define the aggregation functions
    aggregations = {'fv_fm': 'mean', 'fv_fm_end': 'mean'}
    aggregations.update({col: 'mean' for col in y2_cols})
    aggregations.update({'mean_y2': ['mean', 'std'], 'slope_y2': ['mean', 'std']})
    # aggregations.update({'mutated_genes': 'count'}) 

    # Group by 'light_regime' and 'mutated_genes', calculate mean of 'fv_fm' and 'y2' columns
    data_gene = data_slopes.groupby(group_cols, as_index=False).agg(aggregations)

    # Flatten the MultiIndex columns and rename them to match the format
    data_gene.columns = [
        f"{col[0]}_std" if col[1] == 'std' else col[0] 
        for col in data_gene.columns.to_flat_index()
    ]
    # Filter WT genes of plate 99
    wt_plate_99 = data_slopes[(data_slopes['mutant_ID'] == 'WT') & (data_slopes['plate'] == 99)][['light_regime', 'mutated_genes', 'fv_fm', 'fv_fm_end', 'mean_y2', 'slope_y2'] + y2_cols]

    # Group WT genes in sets of three of the same light_regime
    wt_plate_99_grouped = wt_plate_99.groupby('light_regime')

    # Define aggregation functions
    agg_funcs = {col: 'mean' for col in ['fv_fm', 'fv_fm_end'] + y2_cols}
    agg_funcs.update({'mean_y2': ['mean', 'std'], 'slope_y2': ['mean', 'std']})
    # agg_funcs.update({'mutated_genes': 'count'})

    # Group WT genes in sets of three of the same light_regime and aggregate by mean
    wt_plate_99['group'] = wt_plate_99.groupby('light_regime').cumcount() // 3
    artificial_wt = wt_plate_99.groupby(['light_regime', 'group']).agg(agg_funcs).reset_index(level='group', drop=True).reset_index()

    new_columns = []
    for col in artificial_wt.columns.values:
        if col[1] == 'std':
            new_columns.append(f"{col[0]}_std")
        else:
            new_columns.append(col[0])

    artificial_wt.columns = new_columns

    # Rename the 'mutated_genes' of artificial_wt to 'WT_' followed by the group number
    artificial_wt['mutated_genes'] = artificial_wt.groupby('light_regime').cumcount().add(1).astype(str).radd('WT_')

    # put the columns in the same order as gene_pivot
    artificial_wt = artificial_wt[['light_regime', 'mutated_genes', 'fv_fm', 'fv_fm_end', 'mean_y2', 'mean_y2_std', 'slope_y2', 'slope_y2_std'] + y2_cols]

    # Combine with original data_gene
    data_gene = pd.concat([data_gene, artificial_wt], ignore_index=True)

    # Remove 'group' column as it's no longer needed
    data_gene = data_gene.drop(columns=['group'], errors='ignore')
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

def get_norm_data(phase='phase1'):
    data = get_format_data_without_na(phase=phase)
    intra_distance_for_WT = Genes_self_similarity_v2.get_intra_distance_for_WT(data[data['mutant_ID'] == 'WT'])
    thresholds_y2_dict = {}
    thresholds_ynpq_dict = {}
    for light in data['light_regime'].unique():
        pairwise_distances_y2_WT_light = intra_distance_for_WT[(intra_distance_for_WT['light_regime'] == light)]['pairwise_distances_y2'].values
        pairwise_distances_ynpq_WT_light = intra_distance_for_WT[(intra_distance_for_WT['light_regime'] == light)]['pairwise_distances_ynpq'].values
        pairwise_distances_y2_WT_light_flat = [item for sublist in pairwise_distances_y2_WT_light for item in sublist]
        pairwise_distances_ynpq_WT_light_flat = [item for sublist in pairwise_distances_ynpq_WT_light for item in sublist]
        thresholds_y2_dict[light] = np.percentile(pairwise_distances_y2_WT_light_flat, 95)
        thresholds_ynpq_dict[light] = np.percentile(pairwise_distances_ynpq_WT_light_flat, 95)
    data_flagged = Genes_self_similarity_v2.apply_flagging_WT(data, threshold_distance_y2=thresholds_y2_dict, threshold_distance_ynpq=thresholds_ynpq_dict, p=(2/3), threshold_variance = 1)
    data_norm = normalize_data_additive(data_flagged)
    return data_norm

def get_norm_flagged_data(phase='phase1'):
    data = get_format_data_without_na(phase=phase)
    data_WT = data[data['mutant_ID'] == 'WT']
    # mean_intra_distance_for_WT = Genes_self_similarity_v2.get_mean_intra_distance_for_WT(data_WT)
    # weighted_avg_y2 = (mean_intra_distance_for_WT['mean_intra_gene_distance_y2'] * mean_intra_distance_for_WT['sample_count']).sum() / mean_intra_distance_for_WT['sample_count'].sum()
    # weighted_std_y2 = np.sqrt((mean_intra_distance_for_WT['var_intra_gene_distance_y2'] * mean_intra_distance_for_WT['sample_count']).sum() / mean_intra_distance_for_WT['sample_count'].sum())
    # weighted_avg_ynpq = (mean_intra_distance_for_WT['mean_intra_gene_distance_ynpq'] * mean_intra_distance_for_WT['sample_count']).sum() / mean_intra_distance_for_WT['sample_count'].sum()
    # weighted_std_ynpq = np.sqrt((mean_intra_distance_for_WT['var_intra_gene_distance_ynpq'] * mean_intra_distance_for_WT['sample_count']).sum() / mean_intra_distance_for_WT['sample_count'].sum())
    intra_distance_for_WT = Genes_self_similarity_v2.get_intra_distance_for_WT(data[data['mutant_ID'] == 'WT'])
    thresholds_y2_dict = {}
    thresholds_ynpq_dict = {}
    for light in data['light_regime'].unique():
        pairwise_distances_y2_WT_light = intra_distance_for_WT[(intra_distance_for_WT['light_regime'] == light)]['pairwise_distances_y2'].values
        pairwise_distances_ynpq_WT_light = intra_distance_for_WT[(intra_distance_for_WT['light_regime'] == light)]['pairwise_distances_ynpq'].values
        pairwise_distances_y2_WT_light_flat = [item for sublist in pairwise_distances_y2_WT_light for item in sublist]
        pairwise_distances_ynpq_WT_light_flat = [item for sublist in pairwise_distances_ynpq_WT_light for item in sublist]
        thresholds_y2_dict[light] = np.percentile(pairwise_distances_y2_WT_light_flat, 95)
        thresholds_ynpq_dict[light] = np.percentile(pairwise_distances_ynpq_WT_light_flat, 95)
    data_flagged = Genes_self_similarity_v2.apply_flagging_WT(data, threshold_distance_y2=thresholds_y2_dict, threshold_distance_ynpq=thresholds_ynpq_dict, p=(2/3), threshold_variance = 1)
    data_norm = normalize_data_additive(data_flagged)
    data_norm_flagged = Genes_self_similarity_v2.apply_flagging_mutants(data_norm, threshold_distance_y2=thresholds_y2_dict, threshold_distance_ynpq=thresholds_ynpq_dict, p=(2/3), threshold_variance = 1)
    # rename flag_y2_y into flag_y2 and drop flag_y2_x
    data_norm_flagged.rename(columns={'flag_y2_y': 'flag_y2'}, inplace=True)
    data_norm_flagged.rename(columns={'flag_ynpq_y': 'flag_ynpq'}, inplace=True)
    data_norm_flagged.drop(columns='flag_y2_x', inplace=True)
    data_norm_flagged.drop(columns='flag_ynpq_x', inplace=True)
    return data_norm_flagged

def get_smooth_data(data, method='local'):
    data_smooth = data.copy()
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_y2 = data_light.filter(like='y2_').dropna(axis=1).values.astype(float)
        if method == 'local' :
             data_y2_smoothed = np.apply_along_axis(local_smooth, 1, data_y2, d=1, sigma=12)
        elif method == 'kernel' :
            data_y2_smoothed = np.apply_along_axis(kernel_smooth, 1, data_y2, sigma=12)
        # if light == '20h_HL' or light == '20h_ML' or light == '2h-2h' add 40 Nan values at the end
        if light == '20h_HL' or light == '20h_ML' or light == '2h-2h':
            data_y2_smoothed = np.concatenate((data_y2_smoothed, np.full((data_y2_smoothed.shape[0], 40), np.nan)), axis=1)
        
        data_smooth.loc[data['light_regime'] == light, data.filter(like='y2_').columns] = data_y2_smoothed 
    return data_smooth

def get_simple_slope(data):
    features = []
    for i in range(len(data)):
        row = data.iloc[i]
        time_series_y2 = row.filter(regex=r'^y2_\d+$').dropna().values.astype(float)
        time_series_ynpq = row.filter(regex=r'^ynpq_\d+$').dropna().values.astype(float)
        elapsed_time_columns = [col for col in data.columns if 'elapsed_time' in col]
        time = row[elapsed_time_columns].dropna().values[1:-1]
        # get slope of the linear regression on the whole ts
        X_y2 = np.array(time).reshape(-1, 1)
        try:
            model_y2 = LinearRegression().fit(X_y2, time_series_y2)
            slope_y2 = model_y2.coef_[0]
        except:
            print(row['mutant_ID'])
            print(row['light_regime'])
            print(row['plate'])
            print(time)
            print(time_series_y2)
        X_ynpq = np.array(time).reshape(-1, 1)
        model_ynpq = LinearRegression().fit(X_ynpq, time_series_ynpq)
        slope_ynpq = model_ynpq.coef_[0]
        row_features = {
            'slope_y2': slope_y2,
            'slope_ynpq': slope_ynpq
        }
        features.append(row_features)
    features_df = pd.DataFrame(features)
    data_with_features = pd.concat([data.reset_index(drop=True), features_df], axis=1)
    return data_with_features

def get_simple_slope_split(data):
    features = []
    for i in range(len(data)):
        row = data.iloc[i]
        time_series_y2 = row.filter(regex=r'^y2_\d+$').dropna().values.astype(float)
        time_series_ynpq = row.filter(regex=r'^ynpq_\d+$').dropna().values.astype(float)
        elapsed_time_columns = [col for col in data.columns if 'elapsed_time' in col]
        if row['light_regime'] in ['high_2h-2h', 'high_10min-10min', 'high_1min-1min', 'high_30s-30s']:
            time = row[elapsed_time_columns].dropna().values[1:]
        elif row['light_regime'] in ['low_2h-2h', 'low_10min-10min', 'low_1min-1min', 'low_30s-30s']:
            time = row[elapsed_time_columns].dropna().values[:-1]
        else:
            time = row[elapsed_time_columns].dropna().values[1:-1]
        # get slope of the linear regression on the whole ts
        X_y2 = np.array(time).reshape(-1, 1)
        try:
            model_y2 = LinearRegression().fit(X_y2, time_series_y2)
            slope_y2 = model_y2.coef_[0]
        except:
            print(row['mutant_ID'])
            print(row['light_regime'])
            print(time)
            print(time_series_y2)
        X_ynpq = np.array(time).reshape(-1, 1)
        model_ynpq = LinearRegression().fit(X_ynpq, time_series_ynpq)
        slope_ynpq = model_ynpq.coef_[0]
        row_features = {
            'slope_y2': slope_y2,
            'slope_ynpq': slope_ynpq
        }
        features.append(row_features)
    features_df = pd.DataFrame(features)
    data_with_features = pd.concat([data.reset_index(drop=True), features_df], axis=1)
    return data_with_features


def get_pivot_features_mutants(data_slopes):
    grouped = data_slopes.groupby(['mutant_ID', 'plate', 'well_id', 'light_regime']).agg({
        'mean_y2': 'first',
        'slope_y2': 'first',
        'mutated_genes': 'first'
        # 'flag_y2': 'first',
        # 'flag_ynpq': 'first'
    }).reset_index()

    # Step 2: Create combined features using pivot_table
    mutant_data = grouped.pivot_table(index=['mutant_ID', 'plate', 'well_id'], columns='light_regime', values=[
        'mean_y2', 'slope_y2',
    ]).fillna(np.nan)

    # Flatten the multi-index columns
    mutant_data.columns = ['_'.join(map(str, col)).strip() for col in mutant_data.columns.values]

    # Reset index to make 'mutant_ID', 'plate', 'well_ID' regular columns
    mutant_data.reset_index(inplace=True)

    # Merge with 'mutated_genes' and 'GO'
    mutant_data = pd.merge(mutant_data, grouped[['mutant_ID', 'plate', 'well_id', 'mutated_genes']], on=['mutant_ID', 'plate', 'well_id']).drop_duplicates()

    go_terms = data_slopes.groupby(['mutant_ID', 'plate', 'well_id'])['GO'].apply(list).reset_index()

    # agg_go_terms = data_slopes.groupby(['mutant_ID', 'plate', 'well_id'])['aggregated_GO'].apply(list).reset_index()

    mutant_data = pd.merge(mutant_data, go_terms, on=['mutant_ID', 'plate', 'well_id'])
    # mutant_data = pd.merge(mutant_data, agg_go_terms, on=['mutant_ID', 'plate', 'well_id'])

    # flatten the go lists
    # mutant_data['GO'] = mutant_data['GO'].apply(lambda x: [] if x is np.nan else x)
    mutant_data['GO'] = mutant_data['GO'].apply(lambda x: [item for sublist in x for item in sublist])
    # mutant_data['aggregated_GO'] = mutant_data['aggregated_GO'].apply(lambda x: [item for sublist in x for item in sublist])

    mutant_data['GO'] = mutant_data['GO'].apply(lambda x: list(set(x)))
    # mutant_data['aggregated_GO'] = mutant_data['aggregated_GO'].apply(lambda x: list(set(x)))

    return mutant_data

def get_pivot_features_genes(data_gene_y2):
    grouped = data_gene_y2.groupby(['mutated_genes', 'light_regime']).agg({
    'mean_y2': 'first',
    'slope_y2': 'first',
    'mean_y2_std': 'first',
    'slope_y2_std': 'first'
    }).reset_index()

    # Step 2: Create combined features using pivot_table
    gene_data = grouped.pivot_table(index=['mutated_genes'], columns='light_regime', values=[
        'mean_y2', 'slope_y2', 'mean_y2_std', 'slope_y2_std'
    ]).fillna(np.nan)

    # Flatten the multi-index columns
    gene_data.columns = ['_'.join(map(str, col)).strip() for col in gene_data.columns.values]

    # Reset index to make 'mutant_ID', 'plate', 'well_ID' regular columns
    gene_data.reset_index(inplace=True)

    # Merge with 'mutated_genes' and 'GO'
    gene_data = pd.merge(gene_data, grouped[['mutated_genes']], on=['mutated_genes']).drop_duplicates()

    # go_terms = data_gene_y2.groupby(['mutated_genes'])['GO'].apply(list).reset_index()

    # gene_data = pd.merge(gene_data, go_terms, on=['mutated_genes'])

    # flatten the go lists
    # gene_data['GO'] = gene_data['GO'].apply(lambda x: [item for sublist in x for item in sublist])

    return gene_data

def contains_predefined_go_terms(go_list, predefined_list):
    return any(go_term in predefined_list for go_term in go_list)

def get_data_with_certain_GO_expanded(data, go_terms):
    filtered_go_df = data[data['GO'].apply(lambda go_list: contains_predefined_go_terms(go_list, go_terms))]
    filtered_go_df_expanded = filtered_go_df.explode('GO')
    filtered_go_df_expanded = filtered_go_df_expanded[filtered_go_df_expanded['GO'].isin(go_terms)]
    return filtered_go_df_expanded

def get_GO_var(data, go_terms):
    filtered_go_df_expanded = get_data_with_certain_GO_expanded(data, go_terms)

    feature_columns = [col for col in filtered_go_df_expanded.columns if col.startswith('mean') or col.startswith('slope')]

    # Step 2: Filter GO terms with multiple samples
    go_counts = filtered_go_df_expanded['GO'].value_counts()
    go_multiple_samples = go_counts[go_counts > 1].index

    # Step 3: Compute variance for feature columns only for filtered GO terms
    filtered_go_df_expanded = filtered_go_df_expanded[filtered_go_df_expanded['GO'].isin(go_multiple_samples)]
    go_variance = filtered_go_df_expanded.groupby('GO')[feature_columns].var()
    var_filtered_expanded_df = filtered_go_df_expanded[feature_columns].var()

    # Plot a histogram of the variance of each feature for the GO terms
    num_features = len(feature_columns)
    num_rows = (num_features - 1) // 3 + 1

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    colors = plt.cm.get_cmap('tab20', len(go_multiple_samples))

    for i, feature in enumerate(feature_columns):
        ax = axs[i // 3, i % 3]
        variances = go_variance[feature].values
        bins = 30
        color_indices = {go: idx for idx, go in enumerate(go_multiple_samples)}

        n, bins, patches = ax.hist(variances, bins=bins, color='blue', alpha=0.7)
        
        for go in go_multiple_samples:
            variance_value = go_variance.loc[go, feature]
            bin_index = min(int((variance_value - bins[0]) / (bins[1] - bins[0])), bins.size - 2)
            patches[bin_index].set_facecolor(colors(color_indices[go]))
            ax.annotate(go, xy=(variance_value, n[bin_index]), xycoords='data', 
                        xytext=(0, 5), textcoords='offset points', 
                        rotation=90, ha='center', va='bottom', fontsize=8)

        ax.axvline(var_filtered_expanded_df[feature], color='red', linestyle='--')
        ax.axvline(go_variance[feature].mean(), color='green', linestyle='--')
        ax.legend(['Variance of the whole dataset', 'Mean variance of intras GO terms'])
        ax.set_title(feature)
        ax.set_xlabel('Variance')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    return go_variance

def get_GO_mean(data, go_terms):
    filtered_go_df_expanded = get_data_with_certain_GO_expanded(data, go_terms)

    feature_columns = [col for col in filtered_go_df_expanded.columns if col.startswith('mean') or col.startswith('slope')]

    # Step 2: Filter GO terms with multiple samples
    go_counts = filtered_go_df_expanded['GO'].value_counts()
    go_multiple_samples = go_counts[go_counts > 1].index

    # Step 3: Compute mean for feature columns only for filtered GO terms
    filtered_go_df_expanded = filtered_go_df_expanded[filtered_go_df_expanded['GO'].isin(go_multiple_samples)]
    go_means = filtered_go_df_expanded.groupby('GO')[feature_columns].mean()
    mean_filtered_expanded_df = filtered_go_df_expanded[feature_columns].mean()

    # Plot a histogram of the means of each feature for the GO terms
    num_features = len(feature_columns)
    num_rows = (num_features - 1) // 3 + 1

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    colors = plt.cm.get_cmap('tab20', len(go_multiple_samples))

    for i, feature in enumerate(feature_columns):
        ax = axs[i // 3, i % 3]
        means = go_means[feature].values
        bins = 30
        color_indices = {go: idx for idx, go in enumerate(go_multiple_samples)}

        n, bins, patches = ax.hist(means, bins=bins, color='blue', alpha=0.7)
        
        for go in go_multiple_samples:
            mean_value = go_means.loc[go, feature]
            bin_index = min(int((mean_value - bins[0]) / (bins[1] - bins[0])), bins.size - 2)
            patches[bin_index].set_facecolor(colors(color_indices[go]))
            ax.annotate(go, xy=(mean_value, n[bin_index]), xycoords='data', 
                        xytext=(0, 5), textcoords='offset points', 
                        rotation=90, ha='center', va='bottom', fontsize=8)

        ax.axvline(mean_filtered_expanded_df[feature], color='green', linestyle='--')
        ax.legend(['Mean of the whole dataset', 'Mean of intras GO terms'])
        ax.set_title(feature)
        ax.set_xlabel('Mean')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    return go_means

def anova_for_go(data, go_terms):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Filter data for GO terms of interest
    filtered_go_df_expanded = get_data_with_certain_GO_expanded(data, go_terms)

    # List of feature columns
    features = [col for col in filtered_go_df_expanded.columns if col.startswith('mean') or col.startswith('slope')]

    # Initialize list to collect results
    results = []

    # Iterate over each feature
    for feature in features:
        # Rename column to avoid issues with special characters
        renamed_df = filtered_go_df_expanded.rename(columns={feature: feature.replace('-', '_')})
        
        # ANOVA for the renamed feature
        formula = f'{feature.replace("-", "_")} ~ C(GO)'
        model = ols(formula, data=renamed_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extract p-value
        p_value = anova_table["PR(>F)"]["C(GO)"]
        
        # Append result
        results.append({'feature': feature, 'p_value': p_value})

    # Create DataFrame from results
    p_values_df = pd.DataFrame(results)

    return p_values_df

# Define a sorting key function using the order_map
def custom_sort_key(value):
    desired_order = ['20h_HL', '20h_ML', '2h-2h', '10min-10min', '1min-1min', '30s-30s']
    # desired_order = ['20h_HL', '20h_ML', 'low_2h-2h', 'high_2h-2h', 'low_10min-10min', 'high_10min-10min', 'low_1min-1min', 'high_1min-1min', 'low_30s-30s', 'high_30s-30s']
    # Create a dictionary to map each value to its position in the desired order
    order_map = {value: index for index, value in enumerate(desired_order)}
    return order_map.get(value, len(desired_order))

def export_mutants_to_pdf(data_slope_cluster, final_filtered_mutants, filename='plots_mutants.pdf'):
    elapsed_time_columns = [col for col in data_slope_cluster.columns if 'elapsed_time' in col]
    already_seen_genes = []

    with PdfPages(filename) as pdf:
        for i in range(final_filtered_mutants['mutant_ID'].nunique()):
            # mutant_of_interest = result_residuals_slope['mutant_ID'].unique()[i+50]
            # mutant_of_interest = mutant_data_with_pca.sort_values(by='slope_y2_1min-1min')['mutant_ID'].unique()[i]
            mutant_of_interest = final_filtered_mutants.sort_values(by='slope_y2')['mutant_ID'].unique()[i]
            # mutant_of_interest = monotonic_mutants.sort_values(by='slope_y2_1min-1min', ascending=True)['mutant_ID'].unique()[i]
            gene_of_interest = data_slope_cluster[(data_slope_cluster['mutant_ID'] == mutant_of_interest)]['mutated_genes'].unique()[0]
            if gene_of_interest in already_seen_genes or gene_of_interest == 'special_mutant':
                continue
            already_seen_genes.append(gene_of_interest)
            # gene_of_interest = monotonic_mutants.sort_values(by='slope_y2_1min-1min', ascending=True)['mutated_genes'].unique()[i]

            # Sample array containing some of the values
            # light_regimes = data_slopes[(data_slopes['mutant_ID'] == mutant_of_interest)]['light_regime'].unique()
            light_regimes = data_slope_cluster[(data_slope_cluster['mutated_genes'] == gene_of_interest)]['light_regime'].unique()

            # Sort the array using the custom sort key
            sorted_light_regimes = sorted(light_regimes, key=custom_sort_key)
            # titles = ['Y(II) for gene ' + str(gene_of_interest) + ' in ' + light for light in sorted_light_regimes]
            titles = [str(gene_of_interest) + ', ' + light for light in sorted_light_regimes]

            # Prepare subplot layout
            fig, axes = plt.subplots(1, len(titles), figsize=(25, 5))

            # Iterate over light regimes and plot Y(II) values
            for j, (light_regime, title) in enumerate(zip(sorted_light_regimes, titles)):
                # Filter data for the current gene and light regime
                # filtered_data = data_slopes[(data_slopes['mutant_ID'] == mutant_of_interest) & 
                #                                     (data_slopes['light_regime'] == light_regime)]
                filtered_data = data_slope_cluster[(data_slope_cluster['mutated_genes'] == gene_of_interest) & 
                                                    (data_slope_cluster['light_regime'] == light_regime)]
                
                # print('light_regime :' , light_regime, 'fv_fm : ', filtered_data['fv_fm'].values)
                # Extract Y(II) values and plot each line
                y2_values = filtered_data.filter(regex=r'^y2_\d+$').dropna(axis=1).values
                for k in range(y2_values.shape[0]):
                    if light_regime in ['high_2h-2h', 'high_10min-10min', 'high_1min-1min', 'high_30s-30s']:
                        elapsed_time = filtered_data[elapsed_time_columns].dropna(axis=1).values[k][1:]
                    elif light_regime in ['low_2h-2h', 'low_10min-10min', 'low_1min-1min', 'low_30s-30s']:
                        elapsed_time = filtered_data[elapsed_time_columns].dropna(axis=1).values[k][:-1]
                    else:
                        elapsed_time = filtered_data[elapsed_time_columns].dropna(axis=1).values[k][1:-1]
                    if filtered_data['slope_cluster_y2'].values[k] != 0:
                        color = 'r'
                    else:
                        color = 'b'
                    try :
                        axes[j].plot(elapsed_time, y2_values[k], c=color, alpha=0.4, label='Mutant {}'.format(k+1))
                        # plot the linear regression
                        model = LinearRegression()
                        model.fit(elapsed_time.reshape(-1, 1), y2_values[k])
                        y_pred = model.predict(elapsed_time.reshape(-1, 1))
                        axes[j].plot(elapsed_time, y_pred, c='r', alpha=0.4)
                        axes[j].axhline(y=0, color='black', linestyle='--')
                        # Set title and other plot properties
                        axes[j].set_title(title, fontsize=10)
                        axes[j].set_xlabel('Time (h)', fontsize=18)
                        axes[j].set_ylabel('Y(II)', fontsize=18)
                        axes[j].set_ylim(-0.2, 0.2)
                        axes[j].legend(loc='upper right', fontsize='small')
                    except:
                        continue

            # Adjust layout and display the plot
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def plot_mutants_and_genes(data_slope_cluster, final_filtered_mutants, y='y2'):
    elapsed_time_columns = [col for col in data_slope_cluster.columns if 'elapsed_time' in col]
    already_seen_genes = []

    for i in range(min(final_filtered_mutants['mutant_ID'].nunique(), 50)):
    # for i in range(10):
        # gene_of_interest = final_filtered_mutants.index[i]
        # mutant_of_interest = result_residuals_slope['mutant_ID'].unique()[i+50]
        # mutant_of_interest = mutant_data_with_pca.sort_values(by='slope_y2_1min-1min')['mutant_ID'].unique()[i]
        mutant_of_interest = final_filtered_mutants.sort_values(by='slope_y2')['mutant_ID'].unique()[i]
        # mutant_of_interest = final_filtered_mutants.sort_values(by='confidence')['mutant_ID'].unique()[i]
        # mutant_of_interest = monotonic_mutants.sort_values(by='slope_y2_1min-1min', ascending=True)['mutant_ID'].unique()[i]

        try:
            gene_of_interest = data_slope_cluster[(data_slope_cluster['mutant_ID'] == mutant_of_interest)]['mutated_genes'].unique()[0]
        except:
            print('mutant_of_interest : ', mutant_of_interest)
        if gene_of_interest in already_seen_genes or gene_of_interest == 'special_mutant':
            continue
        already_seen_genes.append(gene_of_interest)
        # gene_of_interest = monotonic_mutants.sort_values(by='slope_y2_1min-1min', ascending=True)['mutated_genes'].unique()[i]

        # Sample array containing some of the values
        # light_regimes = data_slopes[(data_slopes['mutant_ID'] == mutant_of_interest)]['light_regime'].unique()
        light_regimes = data_slope_cluster[(data_slope_cluster['mutated_genes'] == gene_of_interest)]['light_regime'].unique()

        # Sort the array using the custom sort key
        sorted_light_regimes = sorted(light_regimes, key=custom_sort_key)
        titles = ['Y(II) for gene ' + str(gene_of_interest) + ' in ' + light for light in sorted_light_regimes]
        # titles = [str(gene_of_interest) + ', ' + light for light in sorted_light_regimes]
        # colors = ['b', 'r', 'g', 'k', 'purple', 'grey']
        # colors = colors[:len(titles)]

        # Prepare subplot layout
        fig, axes = plt.subplots(1, len(titles), figsize=(25, 5))

        # Iterate over light regimes and plot Y(II) values
        for j, (light_regime, title) in enumerate(zip(sorted_light_regimes, titles)):
            # Filter data for the current gene and light regime
            # filtered_data = data_slopes[(data_slopes['mutant_ID'] == mutant_of_interest) & 
            #                                     (data_slopes['light_regime'] == light_regime)]
            filtered_data = data_slope_cluster[(data_slope_cluster['mutated_genes'] == gene_of_interest) & 
                                                (data_slope_cluster['light_regime'] == light_regime)]
            
            # print('light_regime :' , light_regime, 'fv_fm : ', filtered_data['fv_fm'].values)
            # Extract Y(II) values and plot each line
            y2_values = filtered_data.filter(regex=r'^'+y+'_\d+$').dropna(axis=1).values

            for k in range(y2_values.shape[0]):
                if light_regime in ['high_2h-2h', 'high_10min-10min', 'high_1min-1min', 'high_30s-30s']:
                    elapsed_time = filtered_data[elapsed_time_columns].dropna(axis=1).values[k][1:]
                elif light_regime in ['low_2h-2h', 'low_10min-10min', 'low_1min-1min', 'low_30s-30s']:
                    elapsed_time = filtered_data[elapsed_time_columns].dropna(axis=1).values[k][:-1]
                else:
                    elapsed_time = filtered_data[elapsed_time_columns].dropna(axis=1).values[k][1:-1]
                if filtered_data['slope_cluster_' + y].values[k] != 0:
                    color = 'r'
                else:
                    color = 'b'
                # color = 'b'
                try :
                    axes[j].plot(elapsed_time, y2_values[k], c=color, alpha=0.4, label='Mutant {}'.format(k+1))
                    # plot the linear regression
                    model = LinearRegression()
                    model.fit(elapsed_time.reshape(-1, 1), y2_values[k])
                    y_pred = model.predict(elapsed_time.reshape(-1, 1))
                    axes[j].plot(elapsed_time, y_pred, c='r', alpha=0.4)
                    axes[j].axhline(y=0, color='black', linestyle='--')
                    # Set title and other plot properties
                    axes[j].set_title(title, fontsize=10)
                    # axes[j].set_title(light_regime)
                    axes[j].set_xlabel('Time (h)', fontsize=18)
                    axes[j].set_ylabel('Y(II)', fontsize=18)
                    axes[j].set_ylim(-0.2, 0.2)
                    axes[j].legend(loc='upper right', fontsize='small')
                except:
                    continue

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

def int_to_letter(num):
    return chr(ord('A') + num)

def get_mutants_for_Carolyne(final_mutants):
    final_mutants_copy = final_mutants.copy()
    final_mutants_copy['i (converted to letter)'] = final_mutants_copy['i'].apply(int_to_letter)
    final_mutants_copy['j'] = final_mutants_copy['j'] + 1

    final_mutants_for_Carolyne = final_mutants_copy[['mutant_ID', 'mutated_genes', 'plate', 'i (converted to letter)', 'j', 'source_df']]

    return final_mutants_for_Carolyne