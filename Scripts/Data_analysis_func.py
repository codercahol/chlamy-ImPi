import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy
import itertools
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter, MinuteLocator

def plot_y2(data, title='Y(II) Response for Each Mutant', limit=None):
    elapsed_time_columns = [col for col in data.columns if 'elapsed_time' in col]
    for i in range(len(data)):
        row = data.iloc[i]
        if row['light_regime'] in ['high_2h-2h', 'high_10min-10min', 'high_1min-1min', 'high_30s-30s']:
            time = row[elapsed_time_columns].dropna().values[1:]
        elif row['light_regime'] in ['low_2h-2h', 'low_10min-10min', 'low_1min-1min', 'low_30s-30s']:
            time = row[elapsed_time_columns].dropna().values[:-1]
        else:
            time = row[elapsed_time_columns].dropna().values[1:-1]
        data_y2 = row.filter(regex=r'^y2_\d+$').dropna().values
        plt.plot(time, data_y2)
    plt.xlabel('Time')
    plt.ylabel('Y(II) Response')
    if limit != None:
        plt.ylim(limit)
    plt.title(title)
    plt.show()

    # plt.xlabel('Time')
    # plt.ylabel('Y(II) Response')
    # if limit != None:
    #     plt.ylim(limit)
    # plt.title(title)
    # plt.show()

def plot_ynpq(data):
    data_ynpq = data.filter(regex=r'^ynpq_\d+$').dropna(axis=1).values

    for ynpq in data_ynpq:
        plt.plot(ynpq)

    plt.xlabel('Time')
    plt.ylabel('Y(NPQ) Response')
    plt.title('Y(NPQ) Response for Each Mutant')
    plt.show()

def plot_y2_reduced(data, cov):
    data_y2 = data.filter(like='y2_').dropna(axis=1).values
    sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv((3/2)*cov))
    data_y2_reduced = np.dot(data_y2, sqrt_inv)

    for y2 in data_y2_reduced:
        plt.plot(y2)

    plt.xlabel('Time')
    plt.ylabel('Y(II) Response')
    plt.title('Reduced Y(II) Response for Each Mutant')
    plt.show()

def reduce_y2(data, cov):
    data_y2 = data.filter(like='y2_').dropna(axis=1).values
    sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv((3/2)*cov))
    data_y2_reduced = np.dot(data_y2, sqrt_inv)

    return data_y2_reduced

def plot_fv_fm_y2_linear_fit(data, light, ax):
    X = data['fv_fm'].values.reshape(-1, 1)
    y = data['mean_y2'].values.reshape(-1, 1)
    # y = data['y2_1'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    a = data['fv_fm'][data['mutant_ID'] == 'WT'].values.reshape(-1, 1)
    b = data['mean_y2'][data['mutant_ID'] == 'WT'].values.reshape(-1, 1)
    # b = data['y2_1'][data['mutant_ID'] == 'WT'].values.reshape(-1, 1)

    c = data['fv_fm'][data['mutant_ID'] != 'WT'].values.reshape(-1, 1)
    d = data['mean_y2'][data['mutant_ID'] != 'WT'].values.reshape(-1, 1)
    # d = data['y2_1'][data['mutant_ID'] != 'WT'].values.reshape(-1, 1)

    ax.scatter(c, d, color='blue', alpha=0.5, label='Mutants')
    ax.scatter(a, b, color='green', label='WT')
    ax.plot(X, y_pred, color='red', label='Linear Regression')
    # add the rounded values of the intercept and the coefficient to the plot in the bottom-right corner
    ax.text(0.95, 0.05, 'Intercept: ' + str(round(model.intercept_[0], 2)) + '\nCoefficient: ' + str(round(model.coef_[0][0], 2)),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=10)
    # add the legend in the top-left corner
    ax.legend(loc='upper left')
    ax.set_xlabel('fv_fm')
    ax.set_ylabel('Mean of ' + light + ' Y(II) measurements')
    ax.set_title('Mean of ' + light + ' Y(II) measurements as a function of fv_fm')
    ax.set_ylim(-0.05, 0.7)

    return model.intercept_, model.coef_[0][0]

def plot_fv_fm_y2_linear_fit_10min(data, light, ax):
    X = data['fv_fm'].values.reshape(-1, 1)
    # y = data['mean_y2'].values.reshape(-1, 1)
    y = data['y2_2'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    a = data['fv_fm'][data['mutant_ID'] == 'WT'].values.reshape(-1, 1)
    # b = data['mean_y2'][data['mutant_ID'] == 'WT'].values.reshape(-1, 1)
    b = data['y2_2'][data['mutant_ID'] == 'WT'].values.reshape(-1, 1)

    c = data['fv_fm'][data['mutant_ID'] != 'WT'].values.reshape(-1, 1)
    # d = data['mean_y2'][data['mutant_ID'] != 'WT'].values.reshape(-1, 1)
    d = data['y2_2'][data['mutant_ID'] != 'WT'].values.reshape(-1, 1)

    ax.scatter(c, d, color='blue', alpha=0.5, label='Mutants')
    ax.scatter(a, b, color='green', label='WT')
    ax.plot(X, y_pred, color='red', label='Linear Regression')
    # add the rounded values of the intercept and the coefficient to the plot in the bottom-right corner
    ax.text(0.95, 0.05, 'Intercept: ' + str(round(model.intercept_[0], 2)) + '\nCoefficient: ' + str(round(model.coef_[0][0], 2)),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=10)
    # add the legend in the top-left corner
    ax.legend(loc='upper left')
    ax.set_xlabel('fv_fm')
    ax.set_ylabel('Mean of ' + light + ' Y(II) measurements')
    ax.set_title('Mean of ' + light + ' Y(II) measurements as a function of fv_fm')
    ax.set_ylim(-0.05, 0.7)

    return model.intercept_, model.coef_[0][0]

def plot_y2_ynpq(data):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8)) 

    # for each light_regime plot 'mean_y2' and 'mean_ynpq'
    for i, light_regime in enumerate(data['light_regime'].unique()):
        data_light_regime = data[data['light_regime'] == light_regime]
        data_light_regime_WT = data_light_regime[data_light_regime['mutant_ID'] == 'WT']
        # plot 'mean_y2' and 'mean_ynpq'
        ax = axs[i//3, i%3]
        ax.scatter(data_light_regime['mean_y2'], data_light_regime['mean_ynpq'], color='b', label = 'Mutant')
        ax.scatter(data_light_regime_WT['mean_y2'], data_light_regime_WT['mean_ynpq'], color='g', label = 'WT')
        ax.set_title(light_regime)
        ax.set_xlabel('Average Y(II)')
        ax.set_ylabel('Average Y(NPQ)')
        ax.legend()

    plt.tight_layout()
    plt.show()

def variance_per_plate_and_light_regime_y2(data) :
    # get the mean variance of the y_2 columns each plate for each condition
    data_grouped = data.groupby(['plate', 'light_regime'])

    variances = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in data_grouped:
        # Calculate the variance for each 'y2_i' column within the group
        if name[1] == '20h_ML' or name[1] == '20h_HL' or name[1] == '2h-2h':
            group_variances = group.filter(like='y2_').iloc[:, :40].values.var(axis=0)
        else :
            group_variances = group.filter(like='y2_').values.var(axis=0)
        if len(group) > 1 :
            # Compute the mean variance across all 'y2_i' columns for the gene within the light regime
            mean_variance = group_variances.mean()
            # Store the mean variance in the dictionary
            # Convert tuple to string for consistent key format
            key = '#'.join(map(str, name))
            variances[key] = mean_variance

    variance_df = pd.DataFrame(variances.items(), columns=['plate_light_regime', 'mean_variance'])
    variance_df [['plate', 'light_regime']] = variance_df ['plate_light_regime'].str.split('#', n=1, expand=True)
    variance_df.drop(columns=['plate_light_regime'], inplace=True)

    sns.catplot(x='plate', y='mean_variance', hue='light_regime', data=variance_df, kind='bar')
    plt.xlabel('Plate')
    plt.ylabel('Mean Variance')
    plt.title('Mean Variance as a Function of Plate for Each Light Regime')
    plt.show()

    return variance_df

def range_per_plate_and_light_regime_y2(data) :
    # get the mean variance of the y_2 columns each plate for each condition
    data_grouped = data.groupby(['plate', 'light_regime'])

    ranges = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in data_grouped:
        # Calculate the variance for each 'y2_i' column within the group
        if name[1] == '20h_ML' or name[1] == '20h_HL' or name[1] == '2h-2h':
            group_range = group.filter(like='y2_').iloc[:, :40].values.max(axis=1) - group.filter(like='y2_').iloc[:, :40].values.min(axis=1)
        else :
            group_range = group.filter(like='y2_').values.max(axis=1) - group.filter(like='y2_').values.min(axis=1)
        if len(group) > 1 :
            # Compute the mean variance across all 'y2_i' columns for the gene within the light regime
            mean_range = group_range.mean()
            # Store the mean variance in the dictionary
            # Convert tuple to string for consistent key format
            key = '#'.join(map(str, name))
            ranges[key] = mean_range

    range_df = pd.DataFrame(ranges.items(), columns=['plate_light_regime', 'mean_range'])
    range_df [['plate', 'light_regime']] = range_df ['plate_light_regime'].str.split('#', n=1, expand=True)
    range_df.drop(columns=['plate_light_regime'], inplace=True)

    sns.catplot(x='plate', y='mean_range', hue='light_regime', data=range_df, kind='bar')
    plt.xlabel('Plate')
    plt.ylabel('Mean Range')
    plt.title('Mean Range as a Function of Plate for Each Light Regime')
    plt.show()

    return range_df

def variance_per_light_regime_y2(data):
    # get the mean variance of the y_2 columns for each condition
    data_grouped = data.groupby('light_regime')

    variances = {}

    # Iterate over each group and calculate the variance of the 'y2_i' columns
    for name, group in data_grouped:
        # Calculate the variance for each 'y2_i' column within the group
        if name == '20h_ML' or name == '20h_HL' or name == '2h-2h':
            group_variances = group.filter(like='y2_').iloc[:, :40].values.var(axis=0)
        else:
            group_variances = group.filter(like='y2_').values.var(axis=0)
        if len(group) > 1:
            # Compute the mean variance across all 'y2_i' columns for the gene within the light regime
            mean_variance = group_variances.mean()
            # Store the mean variance in the dictionary
            variances[name] = mean_variance

    variance_df = pd.DataFrame(variances.items(), columns=['light_regime', 'mean_variance'])

    sns.catplot(x='light_regime', y='mean_variance', data=variance_df, kind='bar')
    plt.xlabel('Light Regime')
    plt.ylabel('Mean Variance')
    plt.title('Mean Variance as a Function of Light Regime')
    plt.show()

    return variance_df

def get_sample_fv_fm(data):
    df = data[data['light_regime'] == '20h_ML']
    # df_sorted = df.sort_values(by='fv_fm')

    df.reset_index(drop=True, inplace=True)

    num_samples = 31

    # Generate uniformly spaced percentiles to cover the range of fv_fm values
    fv_fm_samples = np.linspace(df['fv_fm'].min(), df['fv_fm'].max(), num_samples)

    # Find the nearest fv_fm values in the DataFrame to the generated samples
    selected_indices = []
    for fv_fm_sample in fv_fm_samples:
        nearest_index = np.abs(df['fv_fm'] - fv_fm_sample).idxmin()
        selected_indices.append(nearest_index)

    samples_df = df.iloc[selected_indices]

    # Fit a linear regression model to the entire dataset
    X = samples_df[['fv_fm']]
    y = samples_df['mean_y2']

    model = LinearRegression()
    model.fit(df[['fv_fm']], df[['mean_y2']])

    # Calculate residuals for each data point
    predictions = model.predict(X)
    all_pred = model.predict(df[['fv_fm']])
    residuals = np.abs(y - predictions.flatten())

    # Define a threshold for residuals (adjust as needed)
    residual_threshold = 0.1  # Example threshold (you can modify this)

    # Select samples that have residuals within the threshold
    selected_df = samples_df[residuals <= residual_threshold]

    # Plot the original data and the selected samples

    plt.figure(figsize=(10, 6))
    plt.scatter(df['fv_fm'], df['mean_y2'], label='Original Data')
    plt.scatter(selected_df['fv_fm'], selected_df['mean_y2'], color='red', label='Selected Samples')
    plt.plot(df['fv_fm'], all_pred, color='black', linestyle='--', label='Linear Regression')
    plt.xlabel('fv_fm')
    plt.ylabel('mean_y2')
    plt.title('Linear Regression with Selected Samples')
    plt.legend()
    plt.grid(True)
    plt.show()

    selected_df = selected_df.drop_duplicates(subset=['fv_fm'])
    selected_df.reset_index(drop=True, inplace=True)

    return selected_df

def get_sample_fv_fm_mean(data):
    # group the data by mutant_ID, average their fv_fm and find their std, keep the well_id and plate
    grouped = data.groupby(['mutant_ID', 'plate', 'well_id']).agg({'fv_fm': ['mean', 'std']}).reset_index()
    grouped.columns = ['mutant_ID', 'plate', 'well_id', 'mean_fv_fm', 'std_fv_fm']
    num_samples = 31

    # Generate uniformly spaced percentiles to cover the range of fv_fm values
    fv_fm_samples = np.linspace(grouped['mean_fv_fm'].min(), grouped['mean_fv_fm'].max(), num_samples)

    selected_indices = []
    for fv_fm_sample in fv_fm_samples:
        nearest_index = np.abs(grouped['mean_fv_fm'] - fv_fm_sample).idxmin()
        selected_indices.append(nearest_index)

    samples_df = grouped.iloc[selected_indices]
    samples_df.drop_duplicates(subset='mean_fv_fm', inplace=True)
    samples_df.reset_index(drop=True, inplace=True)
    return samples_df

def get_fv_fm_ranges_WT(data_WT):
    fv_fm_ranges = data_WT.groupby(['mutant_ID', 'light_regime'])['fv_fm'].max() - data_WT.groupby(['mutant_ID', 'light_regime'])['fv_fm'].min()
    return fv_fm_ranges

def plot_data_for_Liat(data):
    mutant_names = ['CC-4533 (bst4 WT)', 'bst4', 'bst4:BST4', 'bst4 crispr B4', 'bst4 crispr B6']
    data_for_Liat = data[(data['plate'] == 22) & (data['mutant_ID'].isin(mutant_names))]
    data_for_Liat_grouped_light = data_for_Liat.groupby('light_regime')
    colors = ['#6DE6A4', '#CB96F6', 'black', '#999B9D', '#4FA879']
    genes_color = {gene: colors[i] for i, gene in enumerate(data_for_Liat['mutant_ID'].unique())}
    genes_color = {'CC-4533 (bst4 WT)' : 'black',  'bst4' : '#CB96F6', 'bst4:BST4' : '#999B9D', 'bst4 crispr B4' : '#6DE6A4', 'bst4 crispr B6' : '#4FA879'}
    for name, group in data_for_Liat_grouped_light:
        data_for_Liat_grouped_light_gene = group.groupby('mutant_ID')
        data_for_Liat_WT = data[(data['plate'] == 22) & (data['mutant_ID'] == 'WT') & (data['light_regime'] == name)]
        data_for_Liat_WT_y2s = np.mean(data_for_Liat_WT.filter(like='npq_').dropna(axis=1).values, axis=0)
        plt.figure()
        plt.title(name)
        for gene, group_gene in data_for_Liat_grouped_light_gene:
            group_y2s = np.mean(group_gene.filter(like='npq_').dropna(axis=1).values, axis=0)
            t = np.arange(len(group_y2s))
            if name == '1min-1min' or name == '30s-30s':
                # plot one continuous line of the odd values and one for the even one and make them share the same label and color
                plt.plot(t[::2], group_y2s[::2], label=gene, color=genes_color[gene])
                plt.plot(t[1::2], group_y2s[1::2], color=genes_color[gene])
            elif name == '10min-10min': 
                # concat group_y2s[1::4] and group_y2s[2::4]
                # Calculate the length of the resulting array
                total_length_l = len(group_y2s[1::4]) + len(group_y2s[2::4])
                total_length_h = len(group_y2s[::4]) + len(group_y2s[3::4])

                # Initialize the resulting array with zeros
                l = np.zeros(total_length_l)

                # Assign elements from array1 and array2 alternately
                l[0::2] = group_y2s[1::4]
                l[1::2] = group_y2s[2::4]
                # Initialize the resulting array with zeros
                h = np.zeros(total_length_h)

                # Assign elements from array1 and array2 alternately
                h[0::2] = group_y2s[::4]
                h[1::2] = group_y2s[3::4]

                t_l = np.zeros(total_length_l)

                # Assign elements from array1 and array2 alternately
                t_l[0::2] = t[1::4]
                t_l[1::2] = t[2::4]

                t_h = np.zeros(total_length_h)

                # Assign elements from array1 and array2 alternately
                t_h[0::2] = t[::4]
                t_h[1::2] = t[3::4]
                plt.plot(t_l, l, label=gene, color=genes_color[gene])
                plt.plot(t_h, h, color=genes_color[gene])
            else: 
                plt.plot(group_y2s, label=gene, color=genes_color[gene])
        if name == '1min-1min' or name == '30s-30s':
            plt.plot(t[::2], data_for_Liat_WT_y2s[::2], label='Our WT', color='b')
            plt.plot(t[1::2], data_for_Liat_WT_y2s[1::2], color='b')
        elif name == '10min-10min': 
                # concat group_y2s[1::4] and group_y2s[2::4]
                total_length_l = len(data_for_Liat_WT_y2s[1::4]) + len(data_for_Liat_WT_y2s[2::4])
                total_length_h = len(data_for_Liat_WT_y2s[::4]) + len(data_for_Liat_WT_y2s[3::4])

                # Initialize the resulting array with zeros
                l = np.zeros(total_length_l)

                # Assign elements from array1 and array2 alternately
                l[0::2] = data_for_Liat_WT_y2s[1::4]
                l[1::2] = data_for_Liat_WT_y2s[2::4]
                # Initialize the resulting array with zeros
                h = np.zeros(total_length_h)

                # Assign elements from array1 and array2 alternately
                h[0::2] = data_for_Liat_WT_y2s[::4]
                h[1::2] = data_for_Liat_WT_y2s[3::4]

                t_l = np.zeros(total_length_l)

                # Assign elements from array1 and array2 alternately
                t_l[0::2] = t[1::4]
                t_l[1::2] = t[2::4]

                t_h = np.zeros(total_length_h)

                # Assign elements from array1 and array2 alternately
                t_h[0::2] = t[::4]
                t_h[1::2] = t[3::4]
                plt.plot(t_l, l, label='Our WT', color='b')
                plt.plot(t_h, h, color='b')
        else: 
            plt.plot(data_for_Liat_WT_y2s, label='Our WT', color='b')
        plt.xlabel('time')
        plt.ylabel('npq')
        plt.legend()
        plt.show()
    
def get_region(start_letter, end_letter, start_number, end_number) :
    # Generate all possible combinations of letters and numbers within the specified range
    positions = [f"{chr(letter)}{number:02d}" for letter, number in itertools.product(range(ord(start_letter), ord(end_letter) + 1), range(start_number, end_number + 1))]
    return positions

def generate_custom_labels(num_ticks, step=60):
    labels = []
    for i in range(num_ticks):
        minutes = i * step + 30
        hours = minutes // 60
        minutes = minutes % 60
        labels.append(f"{hours}:{minutes:02d}")
        # else:
        #     labels.append(f"{minutes}min")
    return labels

def plot_WT_outliers(data_norm_flagged, outlier, y, light, n):
    outliers = data_norm_flagged[data_norm_flagged['outlier_' + outlier + '_' + y] == True]
    # outliers_div_sorted_y2 = outliers_div_y2.sort_values(by='mean_reduced_v2_y2', ascending=False)
    # plot the WT and the outliers on the same plot
    plt.figure(figsize=(8, 8))
    WT = data_norm_flagged[(data_norm_flagged['light_regime'] == light) & (data_norm_flagged['mutant_ID'] == 'WT') & (data_norm_flagged['flag_' + y] == 'ok')].iloc[:3].filter(like=y + '_').dropna(axis=1).T
    # set the time to a list starting at 0min and increasing by 30min each time and plot this on the x-axis
    times = np.arange(0, 40, 2)
    # plt.plot(WT_y2s, color='blue', alpha=0.3)
    # plt.plot(outliers_mean_sorted[outliers_mean_sorted['light_regime'] == '20h_HL'].filter(like='y2_').dropna(axis=1).iloc[-100:-80].T, color='red', alpha=0.2)
    # plt.plot(times, outliers_mean_sorted_y2[outliers_mean_sorted_y2['light_regime'] == '20h_HL'].filter(like='y2_').dropna(axis=1).iloc[500:600].T, color='red', alpha=0.2)
    outliers_sorted = outliers.sort_values(by='mean_reduced_v2_y2', ascending=False)
    plt.plot(outliers_sorted[outliers_sorted['light_regime'] == light].filter(like=y + '_').dropna(axis=1).iloc[:n].T, color='red', alpha=0.2)
    plt.plot(WT, color='blue', alpha=0.6)
    # add a dashed line at 0
    plt.axhline(y=0, color='black', linestyle='--')
    # plt.plot(times, outliers_mean_sorted_y2[outliers_mean_sorted_y2['light_regime'] == '20h_HL'].filter(like='y2_').dropna(axis=1).iloc[600:700].T, color='red', alpha=0.2)
    # make the xticks rotate
    custom_labels = generate_custom_labels(20)
    plt.xticks(times, custom_labels, rotation=45)
    plt.xlabel('Time (h)', fontsize=18)
    plt.ylim(-0.3, 0.3)
    plt.ylabel('Normalized Y(II)', fontsize=18)
    # plt.title('Y(' + y[1:] + ') values of the WT and the ' +  outliers + ' outliers in the ' + light + ' regime')