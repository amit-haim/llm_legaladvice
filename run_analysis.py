import os
import pandas as pd
from openai import OpenAI
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from datasets import Dataset
from whoosh.index import create_in, open_dir, LockError  # Import LockError
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import openai
import time  # Import the time module
import shutil
import sacrebleu
import seaborn as sns
from pathlib import Path
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import spacy.cli
import statsmodels.api as sm
import ast
import itertools
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import argparse

path = "G:\\My Drive\\Work\\Working Papers\\LLM and Legal Advice"

# Set the project root directory
PROJECT_ROOT = Path(path)

# Function to construct paths relative to the project root
def project_path(*args):
    return PROJECT_ROOT.joinpath(*args)


# OpenAI API setup

# Construct the path to the text file
text_file_path = project_path("scripts", "openai_project_api_key.txt")

# Open and read the text file, saving the content as a string
with open(text_file_path, 'r') as file:
    file_text = file.read()  # Read the entire content into a string

client = OpenAI(
  api_key=file_text,
)






def map_yes_no_to_binary(df, substring):
    """
    Maps all columns containing the specified substring from 'Yes'/'No' to 1/0.
    
    Args:
    - df: The DataFrame containing the columns.
    - substring: The substring to identify the columns to map.
    
    Returns:
    - df: The updated DataFrame with the mapped columns.
    """
    # Select columns that contain the substring
    columns_to_map = [col for col in df.columns if substring in col]
    
    # Define a mapping from 'Yes' to 1 and 'No' to 0
    mapping = {"Yes": 1, "No": 0}
    
    # Apply the mapping to the selected columns
    df[columns_to_map] = df[columns_to_map].replace(mapping)
    
    return df


def plot_means_with_variation(df, variables, variation='std'):
    """
    Creates bar plots showing the mean values and variation for the specified variables,
    using a dictionary for both variable names and their custom labels.
    
    Args:
    - df: The DataFrame containing the data.
    - label_dict: Dictionary mapping variable names (columns) to custom labels.
    - variation: Type of variation to display ('std' for standard deviation, 'sem' for standard error).
    
    Returns:
    - None: Displays the plot.
    """
  # Extract variable names from the dictionary keys
    vars = list()
    for var in variables.keys():
        vars.append(f'{var}_post')
    # Calculate the mean and variation (std or sem)
    means = df[vars].mean()
    
    if variation == 'std':
        errors = df[vars].std()  # Standard deviation
    elif variation == 'sem':
        errors = df[vars].sem()  # Standard error
    else:
        raise ValueError("Invalid variation type. Choose 'std' or 'sem'.")
    
    # Replace the index (variable names) with the custom labels from the dictionary
    custom_labels = [variables[var] for var in variables.keys()]
    
    # Plot the bar plot using matplotlib with error bars
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot with error bars
    x_positions = np.arange(len(means))  # Positions for the bars
    plt.bar(x_positions, means, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    
    # Set custom x-tick labels
    plt.xticks(x_positions, custom_labels, rotation=0)
    
    # Set plot labels and title
    plt.ylabel('Mean Value')
    plt.title(f'Mean and {variation.upper()} of Variables')

    # Set the y-axis range from 1 to 7
    plt.ylim(1, 7)
    
    # Show the plot
    plt.tight_layout()

    plt.savefig(project_path("figures", f"Mean Ranking of Original Post.png"), bbox_inches='tight')
    plt.clf()
    plt.close()
    print("Done with this plot.")

    # plt.show()



def generate_variable_lists(custom_string):
    """
    Generates two lists of variables, with 'contract' replaced by a custom string in the second list (variables_B).
    
    Args:
    - custom_string: The string to plug into variables_B in place of 'contract'.
    
    Returns:
    - variables_A: List of variables for version A.
    - variables_B: List of variables for version B, with the custom string replacing 'contract'.
    """
    # Define the base for variables_A and variables_B without 'contract'
    base_variables = ['use_rank', 'spec_rank', 'fact_rank', 'leg_rank']
    
    # Generate variables_A (static version names)
    variables_A = [f"{base}_post" for base in base_variables]
    
    # Generate variables_B with the custom string replacing 'contract'
    variables_B = [f"{base}_transformed_post_{custom_string}" for base in base_variables]
    
    return variables_A, variables_B


def plot_paired_means_with_variation(df, modified_var, variation='std'):
    """
    Creates grouped bar plots showing the mean values and variation for paired variables
    (e.g., var1_A, var1_B, var2_A, var2_B) in two versions (A and B).
    
    Args:
    - df: The DataFrame containing the data.
    - variables_A: List of variable names for version A (e.g., ['var1_A', 'var2_A']).
    - variables_B: List of variable names for version B (e.g., ['var1_B', 'var2_B']).
    - labels: List of labels for the variables (e.g., ['Variable 1', 'Variable 2']).
    - variation: Type of variation to display ('std' for standard deviation, 'sem' for standard error).
    
    Returns:
    - None: Displays the plot.
    """
    # Labels for the variables
    labels = ['Usability', 'Specificity', 'Factual Robustness', 'Legalistic']

    variables_A, variables_B = generate_variable_lists(modified_var)

    # Calculate the mean and variation for each version
    means_A = df[variables_A].mean()
    means_B = df[variables_B].mean()
    
    if variation == 'std':
        errors_A = df[variables_A].std()  # Standard deviation for version A
        errors_B = df[variables_B].std()  # Standard deviation for version B
    elif variation == 'sem':
        errors_A = df[variables_A].sem()  # Standard error for version A
        errors_B = df[variables_B].sem()  # Standard error for version B
    else:
        raise ValueError("Invalid variation type. Choose 'std' or 'sem'.")
    
    # Set the number of variables
    num_vars = len(variables_A)
    
    # Set positions for the bars on the x-axis
    x_positions = np.arange(num_vars)
    width = 0.35  # Width of the bars
    
    # Plot the bars with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(x_positions - width/2, means_A, width, yerr=errors_A, capsize=5, label='Original Post', color='skyblue', edgecolor='black')
    plt.bar(x_positions + width/2, means_B, width, yerr=errors_B, capsize=5, label=f'Modified Post: {modified_var}', color='lightgreen', edgecolor='black')
    
    # Set the labels and titles
    plt.xticks(x_positions, labels, rotation=0)
    plt.ylabel('Mean Value')
    plt.title(f'Rankings: Original Post vs {modified_var.capitalize()} Modification')
    
    # Add legend
    plt.legend()
    
    # Set y-axis limit to be 1-7 (if applicable, adjust if needed)
    plt.ylim(1, 7)
    
    # Display the plot
    plt.tight_layout()

    plt.savefig(project_path("figures", f"Mean Ranking of original vs {modified_var}.png"), bbox_inches='tight')
    plt.clf()
    print("Done with this plot.")
    # plt.show()

def plot_use_rank_with_modifications(df, label, variation='std'):
    """
    Creates a bar plot showing the mean values and variation for the use_rank variable
    in the baseline and 7 modified versions.
    
    Args:
    - df: The DataFrame containing the data.
    - variation: Type of variation to display ('std' for standard deviation, 'sem' for standard error).
    
    Returns:
    - None: Displays the plot.
    """
    # Define the base variable and the modified versions
    baseline_var = f'{label}_post'
    modified_vars = [f"{label}_transformed_post_{mod}" for mod in var_dict.keys()]
    all_vars = [baseline_var] + modified_vars

    # Calculate the mean and variation for the variables
    means = df[all_vars].mean()
    
    if variation == 'std':
        errors = df[all_vars].std()  # Standard deviation
    elif variation == 'sem':
        errors = df[all_vars].sem()  # Standard error
    else:
        raise ValueError("Invalid variation type. Choose 'std' or 'sem'.")
    
    # Set positions for the bars on the x-axis
    x_positions = np.arange(len(all_vars))
    width = 0.8  # Width of the bars
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, means, width, yerr=errors, capsize=5, color='lightblue', edgecolor='black')
    
    # Set the labels and title
    labels = ['Baseline'] + [var.capitalize() for var in var_dict.keys()]
    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.ylabel('Mean Value')
    plt.title(f'Mean Rankings: {variables[label]} - Baseline vs Modifications')
    
    # Set y-axis limit (adjust if necessary)
    plt.ylim(1, 7)
    
    # Display the plot
    plt.tight_layout()

    plt.savefig(project_path("figures", f"Mean Ranking of {variables[label]} - baseline vs modifications.png"), bbox_inches='tight')
    plt.close()
    plt.clf()
    print("Done with this plot.")
    # plt.show()

# Download the model if not already installed
spacy.cli.download("en_core_web_md")

# Load a spaCy model with vectors (you can use 'en_core_web_md' or 'en_core_web_trf')
nlp = spacy.load("en_core_web_md")

# Function to convert a single text to its spaCy vector representation
def get_spacy_embedding(text):
    return nlp(text).vector

def calculate_embeddings(df,var):
    # Apply the embedding function to columns
    df['embedding_original'] = df['best_response_post'].apply(get_spacy_embedding)
    df[f'embedding_{var}'] = df[f'best_response_transformed_post_{var}'].apply(get_spacy_embedding)

    # Calculate cosine similarity for each pair of embeddings
    df['cosine_similarity_answer'] = df.apply(lambda row: cosine_similarity(
        [row['embedding_original']], [row[f'embedding_{var}']])[0][0], axis=1)

    return df

def generate_plot_cosine(df,modified_var):
    df = calculate_embeddings(df,modified_var)
    
    overall_mean = df['cosine_similarity_answer'].mean()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cosine_similarity_answer'], kde=True, bins=50)
    plt.axvline(overall_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {overall_mean:.2f}')
    plt.title(f'Distribution of cosine similarity between answers with original and modified post')
    plt.xlabel('Mean Value')
    plt.ylabel('Density')
    plt.savefig(project_path("figures", f"cosine similarity between answers of original vs {modified_var}.png"), bbox_inches='tight')
    plt.close()
    plt.clf()
    print("Done with this plot.")

def generate_mean_cosine_plot(df):
    # Store the means and standard deviations
    cosine_means = []
    cosine_stddevs = []
    variables = []
    
    # Calculate cosine similarity for each variable and store the mean and std dev
    for var in var_dict.keys():
        df = calculate_embeddings(df, var)
        mean_cosine = df['cosine_similarity_answer'].mean()
        stddev_cosine = df['cosine_similarity_answer'].std()
        
        cosine_means.append(mean_cosine)
        cosine_stddevs.append(stddev_cosine)
        variables.append(var)
    
    # Create the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(variables, cosine_means, yerr=cosine_stddevs, capsize=5, color='skyblue')
    plt.xlabel('Variables')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('Mean Cosine Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.ylim(0.9, 1.01)
    
    # Save the plot to a file
    plt.savefig(project_path("figures", "mean_cosine_similarity_bar_plot.png"), bbox_inches='tight')
    plt.clf()
    print("Done with this plot.")


def run_regression(df, col):
    list = [f'{col}_post']
    for var in var_dict.keys():
        list.append(f'{col}_transformed_post_{var}')
        
    df1 = df[list]
    
    # Reshape the data from wide to long format
    df_long = pd.melt(df1, var_name='transformation', value_name=col, ignore_index=False)
    
    # Map the baseline 'use_rank_post' to the baseline group
    df_long['baseline'] = df_long['transformation'].apply(lambda x: 1 if x == f'{col}_post' else 0)
    
    # Create dummy variables for the transformations, leaving the baseline as the reference
    df_long['is_jurisdiction'] = df_long['transformation'].apply(lambda x: 1 if 'jurisdiction' in x else 0)
    df_long['is_dates'] = df_long['transformation'].apply(lambda x: 1 if 'dates' in x else 0)
    df_long['is_money'] = df_long['transformation'].apply(lambda x: 1 if 'money' in x else 0)
    df_long['is_laws'] = df_long['transformation'].apply(lambda x: 1 if 'laws' in x else 0)
    df_long['is_contract'] = df_long['transformation'].apply(lambda x: 1 if 'contract' in x else 0)
    df_long['is_documentation'] = df_long['transformation'].apply(lambda x: 1 if 'documentation' in x else 0)
    df_long['is_numbers'] = df_long['transformation'].apply(lambda x: 1 if 'numbers' in x else 0)
    
    # Ensure all dummy variables and 'use_rank' are numeric
    X = df_long[['is_jurisdiction', 'is_dates', 'is_money', 'is_laws', 'is_contract', 'is_documentation', 'is_numbers']]
    X = sm.add_constant(X)  # Add an intercept
    y = df_long[col]
    
    # Fit the regression model
    model = sm.OLS(y, X)
    results = model.fit()

    # Display the regression results in LaTeX format
    latex_table = results.summary2().as_latex()

    save_dir = os.path.join("G:", os.sep, "My Drive", "Work", "Working Papers", "LLM and Legal Advice", "figures")

    # Specify the output file name based on the column name
    output_file = os.path.join(save_dir,f'regression_results_{col}.txt')
    
    # Save the LaTeX table to the specified file
    with open(output_file, 'w') as f:
        f.write(latex_table)
        
    return results


# Function to plot the coefficients
def plot_regression_coefficients(results,col):
    """
    Plots the regression coefficients from a statsmodels OLS regression result.
    
    Args:
    - results: The fitted OLS model result.
    
    Returns:
    - None: Displays the plot.
    """
    # Get the coefficients and standard errors
    coefficients = results.params
    errors = results.bse  # Standard errors
    
    # Get the names of the coefficients (excluding the intercept)
    labels = coefficients.index[1:]  # Exclude the intercept
    coeff_values = coefficients[1:]  # Exclude the intercept coefficient
    error_values = errors[1:]  # Exclude the intercept error

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, coeff_values, yerr=error_values, capsize=5, color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Transformation')
    plt.ylabel('Coefficient Value')
    plt.title(f'Regression Coefficients for {col}')
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Rotate the x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(project_path("figures", f"regression_coefs_for_{col}.png"), bbox_inches='tight')
    plt.close()
    plt.clf()
    print("Done with this plot.")

def regressions(df,col):
    results = run_regression(df,col)
    plot_regression_coefficients(results,col)




# Create a big list from a DataFrame column containing lists or strings
def create_list(df, var):
    missing_vars_list = []
    
    for item in df[var]:
        # Try to parse strings as lists
        if isinstance(item, str):
            try:
                item = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                pass  # Keep as a string if it's not a list
        
        # Check the type using type()
        if isinstance(item, list):
            missing_vars_list.extend(item)
        else:
            missing_vars_list.append(item)
    
    # Flatten the list of lists into one big list
    missing_vars_list = list(itertools.chain.from_iterable(
        [item if isinstance(item, list) else [item] for item in missing_vars_list]))
    
    # Split any items containing '@' into separate items
    final_list = []
    for item in missing_vars_list:
        if isinstance(item, str):
            split_items = [sub_item.strip() for sub_item in item.split('@') if sub_item]
            final_list.extend(split_items)
    return final_list

# Classify categories using Hugging Face zero-shot classification
def classify_categories(final_list):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = list(var_dict.keys())
    
    classified_items = {}
    for item in final_list:
        result = classifier(item, candidate_labels)
        top_labels = result['labels'][:2]  # Get top 2 labels
        classified_items[item] = top_labels
    return classified_items

# Plot categories side-by-side for two variables
def plot_categories_side_by_side(df, var_name1, var_name2):
    var1 = f'var_missing_{var_name1}_list'
    var2 = f'var_missing_transformed_{var_name1}_{var_name2}_list'

    # Get lists and classify categories for both variables
    final_list1 = create_list(df, var1)
    classified_items1 = classify_categories(final_list1)
    
    final_list2 = create_list(df, var2)
    classified_items2 = classify_categories(final_list2)
    
    # Count category frequencies for both variables
    category_counts1 = Counter()
    for categories in classified_items1.values():
        category_counts1.update(categories)
    
    category_counts2 = Counter()
    for categories in classified_items2.values():
        category_counts2.update(categories)
    
    # Ensure both sets of categories are aligned
    all_categories = list(set(list(category_counts1.keys()) + list(category_counts2.keys())))
    counts1 = [category_counts1.get(cat, 0) for cat in all_categories]
    counts2 = [category_counts2.get(cat, 0) for cat in all_categories]
    
    # Plot side-by-side bar chart
    bar_width = 0.35
    index = np.arange(len(all_categories))

    plt.figure(figsize=(12, 7))
    
    # Bars for var1 with custom label for legend
    plt.bar(index, counts1, bar_width, label='Original Post', color='skyblue')
    
    # Bars for var2 with custom label for legend
    plt.bar(index + bar_width, counts2, bar_width, label=f'Modified Post: {var_name2.capitalize()}', color='salmon')
    
    # Labels and title
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Classified Categories: Original {var_name1.capitalize()} vs {var_name2.capitalize()} Modification')
    plt.xticks(index + bar_width / 2, all_categories, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Ensure directory exists and save the plot
    save_dir = os.path.join("G:", os.sep, "My Drive", "Work", "Working Papers", "LLM and Legal Advice", "figures")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Frequency of Classified Categories original {var_name1} vs {var_name2}.png"), bbox_inches='tight')
    print("Done with this plot.")


# Function to calculate standard error for frequency counts
def calculate_standard_error(counts):
    return np.sqrt(counts) / np.sqrt(len(counts))  # Simplified assumption for error calculation

# Plot two bars per variable: Original Post vs Corresponding Modification with calculated error bars
def plot_baseline_vs_modifications_with_calculated_errors(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.35  # Adjust bar width for two bars per variable
    x_positions = np.arange(len(var_dict))  # Position for each variable

    # Loop through each variable (modification) and plot against the baseline
    for i, (key, desc) in enumerate(var_dict.items()):
        var1 = 'var_missing_post_list'
        var2 = f'var_missing_transformed_post_{key}_list'
        
        # Get classified categories for the baseline (original post)
        final_list1 = create_list(df, var1)
        classified_items1 = classify_categories(final_list1)
        category_counts1 = Counter()
        for categories in classified_items1.values():
            category_counts1.update(categories)
        
        # Get classified categories for the corresponding modification
        final_list2 = create_list(df, var2)
        classified_items2 = classify_categories(final_list2)
        category_counts2 = Counter()
        for categories in classified_items2.values():
            category_counts2.update(categories)
        
        # Only plot the counts for the current variable being modified
        count_baseline = category_counts1.get(key, 0)
        count_modification = category_counts2.get(key, 0)

        # Calculate standard error for both baseline and modification
        error_baseline = calculate_standard_error([count_baseline])
        error_modification = calculate_standard_error([count_modification])

        # Plot bars with calculated error bars for baseline and corresponding modification
        ax.bar(x_positions[i], count_baseline, bar_width, color='skyblue', yerr=error_baseline, capsize=5)
        ax.bar(x_positions[i] + bar_width, count_modification, bar_width, color='lightgreen', yerr=error_modification, capsize=5)

    # Set labels, title, and formatting for the plot
    ax.set_xlabel('Categories')
    ax.set_ylabel('Frequency')
    ax.set_title('Category Frequency: Original Post vs Modifications')
    ax.set_xticks(x_positions + bar_width / 2)
    ax.set_xticklabels(var_dict.keys(), rotation=45, ha='right')
    
    # Custom legend with skyblue for Original Post and green for Modification
    ax.legend(['Original Post', 'Modification'], loc='upper right')

    plt.tight_layout()

    # Ensure directory exists and save the plot
    save_dir = os.path.join("G:", os.sep, "My Drive", "Work", "Working Papers", "LLM and Legal Advice", "figures")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Frequency_of_Classified_Categories_Post_vs_Modifications_Calculated_Errors.png"), bbox_inches='tight')
    plt.close()
    plt.clf()
    print("Done with this plot.")

def run_analysis(df_answers):
    error_log = []  # Initialize an empty list to store error messages
    
    # Try block for generate_mean_cosine_plot
    try:
        generate_mean_cosine_plot(df_answers)
    except Exception as e:
        error_log.append(f"Error in function 'generate_mean_cosine_plot': {e}")
    
    # Loop for generate_plot_cosine
    for var in var_dict:
        try:
            generate_plot_cosine(df_answers, var)
        except Exception as e:
            error_log.append(f"Error in function 'generate_plot_cosine' with var '{var}': {e}")
    
    # Loop for plot_use_rank_with_modifications
    for label in variables.keys():
        try:
            plot_use_rank_with_modifications(df_answers, label, variation='std')
        except Exception as e:
            error_log.append(f"Error in function 'plot_use_rank_with_modifications' with label '{label}': {e}")
    
    # Try block for plot_means_with_variation
    try:
        plot_means_with_variation(df_answers, variables, variation='std')
    except Exception as e:
        error_log.append(f"Error in function 'plot_means_with_variation': {e}")
    
    # Loop for plot_paired_means_with_variation
    for var in var_dict.keys():
        try:
            plot_paired_means_with_variation(df_answers, var, variation='std')
        except Exception as e:
            error_log.append(f"Error in function 'plot_paired_means_with_variation' with var '{var}': {e}")
    
    # Loop for regressions
    for col in variables.keys():
        try:
            regressions(df_answers, col)
        except Exception as e:
            error_log.append(f"Error in function 'regressions' with col '{col}': {e}")

 # Try block for plot_baseline_vs_modifications_with_calculated_errors
    try:
        plot_baseline_vs_modifications_with_calculated_errors(df_answers)
    except Exception as e:
        error_log.append(f"Error in function 'plot_baseline_vs_modifications_with_calculated_errors': {e}")
        
    # Loop for plot_categories_side_by_side
    for key in var_dict.keys():
        try:
            plot_categories_side_by_side(df_answers, 'post', key)
        except Exception as e:
            error_log.append(f"Error in function 'plot_categories_side_by_side' with key '{key}': {e}")

   
    # After all function calls, check if there are any errors in the log
    if error_log:
        print("Errors encountered during execution:")
        for error in error_log:
            print(error)
    else:
        print("All functions executed successfully.")


var_dict = {
    "jurisdiction": "may inform about the jurisdiction, such as the state or city",
    "dates": "is about specific dates, years, or periods of time",
    "money": "is about sums of money",
    "laws": "is about specific laws or regulations, including their names, references, or language",
    "contract": "is about specific lease and contract language",
    "documentation": "is related to documentation of communications, such as keeping emails, texts, recording phone calls, including the existence of documentation",
    "numbers": "contains numbers or measures, such as sizes, distances or heights (excluding dates or money)"
}

# Dictionary for variable names and custom labels
variables = {
    'use_rank': 'Usability',
    'spec_rank': 'Specificity',
    'fact_rank': 'Factual Robustness',
    'leg_rank': 'Legalistic'
}


# Sampling function: Use the sample size provided
def sample_dataframe(df, num_samples):
    """
    Samples the DataFrame if num_samples is specified.
    
    Args:
    - df: The DataFrame to sample from.
    - num_samples: Number of rows to sample.
    
    Returns:
    - Sampled DataFrame or original DataFrame if num_samples is None.
    """
    if num_samples is not None and num_samples > 0:
        df = df.sample(n=num_samples, random_state=42)
    return df

def main(num_samples):
    # Load the dataset
    df_answers = pd.read_csv(project_path("data", "df_answers_full.csv"))
    df_answers.columns = df_answers.columns.str.lower()

    # Sample the dataset based on the argument
    df_answers = sample_dataframe(df_answers, num_samples)

    # Run your processing here (e.g., map_yes_no_to_binary, run_analysis, etc.)
    df_answers = map_yes_no_to_binary(df_answers, substring="legal_prob_")
    df_answers = map_yes_no_to_binary(df_answers, substring="legal_info_")

    print("starting analysis")
    # Call the rest of your analysis functions
    run_analysis(df_answers)
    
    print("All done!")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run analysis on a sampled dataset.")
    
    # Add argument for the number of rows to sample
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,  # If not specified, no sampling will be done.
        help="Number of rows to sample from the dataset (default: entire dataset)."
    )
    
    # Parse the arguments
    args = parser.parse_args()

    # Pass the number of samples to the main function
    main(args.num_samples)
