import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import random
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from factor_analyzer import FactorAnalyzer
import seaborn as sns
from scipy.stats import pearsonr
import scipy.stats as stats
import statsmodels.api as sm

def rChart(x, y, xname, yname, noise=False, u=0, sd=1, size=5, lowessReg=True):
    # Calculate the correlation coefficient and the p-value
    r, p = stats.pearsonr(x, y)

    # Calculate the sample size
    n = len(x)


    # Compute the 95% confidence interval for r
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-(1-0.95)/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    if (lowessReg == True):
        # Add the LOESS regression line
        lowess = sm.nonparametric.lowess
        z = lowess(y, x, frac=1./3, it=0)  # you can modify frac and it as needed
        plt.plot(z[:, 0], z[:, 1], color='blue')

    # Add the line of best fit
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red')
    
    if(noise==True):
        noise1 = np.random.normal(loc=u, scale=sd, size=len(x))
        noise2 = np.random.normal(loc=u, scale=sd, size=len(x))
        plt.scatter(x + noise1, y + noise2, s=size, color="black")     
    else:
        # Create the scatter plot
        plt.scatter(x, y, s=size, color="black")


    # Add the correlation coefficient, 95% CI, and sample size to the plot
    plt.text(0.05, 1.02, f'r = {r:.2f}, 95% CI = [{lo:.2f}, {hi:.2f}], n = {n}', transform=plt.gca().transAxes)
    # Add labels and title
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Display the plot
    plt.show()


def histograms(x, y, xname, yname, varname):
    mean_0 = x.mean()
    std_0 = y.std()

    mean_1 = y.mean()
    std_1 = y.std()
    d = (mean_0 - mean_1) / ( (std_0 + std_1)/2)
    t_stat, p_value = stats.ttest_ind(x, y)
    # Create a histogram for the group where binary_col is 0
    x.hist(alpha=0.5, bins=30, label='1. ' + xname)

    # Create a histogram for the group where binary_col is 1
    y.hist(alpha=0.5, bins=30, label='2. ' + yname)

    # Add labels
    plt.xlabel(varname)
    plt.ylabel('Frequency')
    plt.text(0.68, 0.80, f'd = {d:.3f}, p = {p_value:.3f}', transform=plt.gca().transAxes)
    plt.text(0.68, 0.75, r'$\mu_1$ = {:.2f}, $\mu_2$ = {:.2f}'.format(mean_0, mean_1), transform=plt.gca().transAxes)
    plt.text(0.68, 0.70, r'$n_1$ = {}, $n_2$ = {}'.format(len(x), len(y)), transform=plt.gca().transAxes)
    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

def histosim(xname='X', yname='Y', varname='VAR', xStats=(0,1), yStats=(0,1), xn=1000, yn=1000, fileName='file.png'):
    meanx, sdx = xStats
    meany, sdy = yStats
    # Simulate the data based on Gaussian distribution
    x = np.random.normal(meanx, sdx, xn)
    y = np.random.normal(meany, sdy, yn)

    mean_0 = x.mean()
    std_0 = x.std()

    mean_1 = y.mean()
    std_1 = y.std()
    
    d = (meanx - meany) / ( (sdx + sdy)/2)
    t_stat = d * np.sqrt( (xn*yn)/(xn+yn) )
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), xn+yn-2))
    
    # Create a histogram for the group x
    plt.hist(x, alpha=0.5, bins=30, label='1. ' + xname)

    # Create a histogram for the group y
    plt.hist(y, alpha=0.5, bins=30, label='2. ' + yname)

    # Add labels
    plt.xlabel(varname)
    plt.ylabel('Frequency')
    plt.text(0.68, 0.80, f'd = {d:.3f}, p = {p_value:.3f}', transform=plt.gca().transAxes)
    plt.text(0.68, 0.75, r'$\mu_1$ = {:.2f}, $\mu_2$ = {:.2f}'.format(meanx, meany), transform=plt.gca().transAxes)
    if (xn != 1000 or yn != 1000):
        plt.text(0.68, 0.70, r'$n_1$ = {}, $n_2$ = {}'.format(len(x), len(y)), transform=plt.gca().transAxes)
    
    # Add a legend
    plt.legend()
    plt.savefig(fileName, dpi=300)  # Save the figure as a high-res PNG file

    # Display the plot
    plt.show()

def histosim_multi(group_names, group_stats, varname='VAR', n=1000, fileName='file.png'):
    # Example usage:
    # histosim_multi(['Group 1', 'Group 2'], [(0, 1), (2, 1.5)], varname='Example Variable')
    # Check if the input lists are of the same length
    if len(group_names) != len(group_stats):
        raise ValueError("The length of group_names and group_stats must be the same.")

    # Simulate the data and plot histograms for each group
    for i, (name, stats) in enumerate(zip(group_names, group_stats)):
        mean, sd = stats
        data = np.random.normal(mean, sd, n)
        plt.hist(data, alpha=0.5, bins=30, label=f'{i+1}. {name}', linestyle='-', linewidth=2, histtype='step')

        # Display mean and std dev on the plot
        plt.text(0.68, 0.90 - i*0.05, r'$\mu_{}$ = {:.2f}, $\sigma_{}$ = {:.2f}'.format(i+1, mean, i+1, sd ), transform=plt.gca().transAxes)

    # Add labels
    plt.xlabel(varname)
    plt.ylabel('Frequency')
    
    # Add a legend on the left outside of the plot
    plt.legend(loc='upper left')

    # Save the figure as a high-res PNG file
    plt.savefig(fileName, dpi=300)

    # Display the plot
    plt.show()



def groupRchart(df, groupby, group, names=None, title='Plot', n=1, exclude=None):
    #df is the dataframe with the data
    #groupby are the things to clump eg ages
    #group is the thing to get the mean of by each groupby. Eg the mean IQ by age
    #names are a tuple if you want to change axis names. (xname, yname)
    #title changes plot title
    #n groups by every n of groupby. So 2 would be it groups by every 2 years. 10 every decade etc. 
    #exclude is a list of values to exclude eg [10, 20, 30, 50]
    xname = groupby
    yname = group
    if (names != None):
        xname, yname = names
    # 1. Create a new column that bins the age into decades
    df['decade'] = (df[groupby] // n) * n
    if exclude is not None:
        df = df[~df['decade'].isin(exclude)]

    # 2. Group by the new decade column and compute the mean of the binary column
    df_decade = df.groupby('decade')[group].agg(['mean', 'sem']).reset_index()

    # Calculate the margins (1.96 * standard error)
    df_decade['lower'] = df_decade['mean'] - 1.96 * df_decade['sem']
    df_decade['upper'] = df_decade['mean'] + 1.96 * df_decade['sem']

    # Calculate line of best fit
    slope, intercept = np.polyfit(df_decade['decade'], df_decade['mean'], 1)

    # Calculate correlation coefficient (r value)
    r_value = np.corrcoef(df_decade['decade'], df_decade['mean'])[0, 1]
    
    # Compute the 95% confidence interval for r
    r_z = np.arctanh(r_value)
    se = 1/np.sqrt(df_decade['decade'].size-3)
    z = stats.norm.ppf(1-(1-0.95)/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))

    # Plot the means by decade in a scatter plot with error bars
    plt.errorbar(df_decade['decade'], df_decade['mean'], yerr=(df_decade['mean']-df_decade['lower'], df_decade['upper']-df_decade['mean']), fmt='o')
    plt.plot(df_decade['decade'], slope * df_decade['decade'] + intercept, color='red')  # add line of best fit
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)

    # Add r value and equation to the plot
    plt.text(0.05, 0.95, f'r = {r_value:.2f}, 95% CI = [{lo:.2f}, {hi:.2f}]', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'y = {slope:.2f}x + {intercept:.2f}', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()

def groupRchart2(data, groupby, group, explicit=True, names=None, title='Plot', n=1, exclude=None, labels=None):
    #df is the dataframe with the data
    #groupby are the things to clump eg ages
    #group is the thing to get the mean of by each groupby. Eg the mean IQ by age
    #names are a tuple if you want to change axis names. (xname, yname)
    #title changes plot title
    #n groups by every n of groupby. So 2 would be it groups by every 2 years. 10 every decade etc. 
    #exclude is a list of values to exclude eg [10, 20, 30, 50]
    xname = groupby
    yname = group
    colors = ['blue', 'orange']
    if (labels is None):
        labels = ['<35 fathers', '>35 fathers']
    if (names != None):
        xname, yname = names
    if data is not None:
        for df, c, l in zip(data, colors, labels):
            # 1. Create a new column that bins the age into decades
            df['decade'] = (df[groupby] // n) * n
            if exclude is not None:
                df = df[~df['decade'].isin(exclude)]

            # 2. Group by the new decade column and compute the mean of the binary column
            df_decade = df.groupby('decade')[group].agg(['mean', 'sem']).reset_index()

            # Calculate the margins (1.96 * standard error)
            df_decade['lower'] = df_decade['mean'] - 1.96 * df_decade['sem']
            df_decade['upper'] = df_decade['mean'] + 1.96 * df_decade['sem']

            # Calculate line of best fit
            slope, intercept = np.polyfit(df_decade['decade'], df_decade['mean'], 1)

            # Calculate correlation coefficient (r value)
            r_value = np.corrcoef(df_decade['decade'], df_decade['mean'])[0, 1]

            # Compute the 95% confidence interval for r
            r_z = np.arctanh(r_value)
            se = 1/np.sqrt(df_decade['decade'].size-3)
            z = stats.norm.ppf(1-(1-0.95)/2)
            lo_z, hi_z = r_z-z*se, r_z+z*se
            lo, hi = np.tanh((lo_z, hi_z))

            # Plot the means by decade in a scatter plot with error bars
            if (explicit is True):
                plt.errorbar(df_decade['decade'], df_decade['mean'], yerr=(df_decade['mean']-df_decade['lower'], df_decade['upper']-df_decade['mean']), fmt='o', label=l, color=c)
            plt.plot(df_decade['decade'], slope * df_decade['decade'] + intercept,)  # add line of best fit
            plt.xlabel(xname)
            plt.ylabel(yname)
            plt.title(title)
            
    plt.legend()  # add this line

    plt.grid(True)
    plt.show()

def barChart(data, yAxis = 'Y Axis', title='Plot Title', fileName='plot.png', dpi=300, yStart=None, labelAngle=0):
    # data is a dict where key is the item label, and val is (mean, se)
    # Plots mean with 95% CI error bars
    defaultColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    labels = list(data.keys())
    means = [val[0] for val in data.values()]
    standard_errors = [val[1] for val in data.values()]
    colors = []
    for i, val in enumerate(data.values()):
        if (len(val) > 2):
            colors.append(val[2])
        else:
            colors.append(defaultColors[i%9])

    # Convert standard errors to 95% confidence interval error bars
    confidence_interval = [se * 1.96 for se in standard_errors]

    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    
    for i, (mean, ci, color) in enumerate(zip(means, confidence_interval, colors)):
        ax.bar(i, mean, align='center', alpha=0.5, color=color)
        if ci != 0:  # Only plot the error bar if ci is not 0
            ax.errorbar(i, mean, yerr=ci, ecolor='black', capsize=10, fmt='none')

    ax.set_ylabel(yAxis)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=labelAngle)
    ax.set_title(title)
    ax.yaxis.grid(True)
    if yStart is not None:
        ax.set_ylim(bottom=yStart)

    plt.tight_layout()
    plt.savefig(fileName, dpi=dpi)  # Save the figure as a high-res PNG file
    plt.show()

def sdToSE(sd, n):
    return sd / (n**0.5)

def newLineCite(s, length=25):
    # Remove existing newline characters
    s = s.replace('\n', '')
    s = s.replace('\t', '')
    # Initialize variables
    new_string = ""
    last_cut = 0

    # Iterate over the string in steps of 'length'
    for i in range(length, len(s) + length, length):
        # Find the nearest space before the cut-off point
        cut_point = s.rfind(' ', last_cut, i)

        # If no space is found, use the cut-off point
        if cut_point == -1:
            cut_point = min(i, len(s))

        # Add the substring to the new string
        new_string += s[last_cut:cut_point].strip() + '\n'
        last_cut = cut_point + 1

    # Remove the last newline character if it's at the end of the string
    return new_string.rstrip()


def rChartSim(xLabel, yLabel, rValue, mean=(0,0), sd=(1,1), n=1000, size=5, fileName='plot.png', title='Scatter Plot', floor=None, ceil=None, citation='', textloc=None):
    # floor and ceil should be a tuple
    # Generate random data
    cov_matrix = [[sd[0]**2, rValue*sd[0]*sd[1]], [rValue*sd[0]*sd[1], sd[1]**2]]  # covariance matrix
    data = np.random.multivariate_normal(mean, cov_matrix, size=n)
    x, y = data[:, 0], data[:, 1]

    # If floor is set, remove data points below the floor
    if floor is not None:
        mask = (x > floor[0]) & (y > floor[1])  # only keep values above the floor
        x, y = x[mask], y[mask]

    if ceil is not None:
        mask = (x < ceil[0]) & (y < ceil[1])  # only keep values above the floor
        x, y = x[mask], y[mask]

    # Calculate regression line
    m, b = np.polyfit(x, y, 1)
    reg_x = np.linspace(min(x), max(x), 100)
    reg_y = m * reg_x + b

    # Create plot
    plt.scatter(x, y, s=size)
    plt.plot(reg_x, reg_y, color='red')  # regression line
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    corner = lambda r : 0.15 if r > 0 else 0.95
    plt.text(corner(rValue), 0.95, f'r={rValue:.2f}', ha='right', va='top', transform=plt.gca().transAxes)
    # Adding citation text at the bottom right corner
    if (textloc is None):
        corner2 = lambda r: 0.95 if r > 0 else 0.30
    else:
        corner2 = lambda r: textloc
    plt.text(corner2(rValue), 0.05, citation, ha='right', va='bottom', fontsize=8, transform=plt.gca().transAxes)

    plt.savefig(fileName, dpi=300)  # Save the figure as a high-res PNG file
    plt.show()
