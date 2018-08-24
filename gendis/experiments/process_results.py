# Process result files in directory and create summarizing statistics + plots

DIRECTORY = 'results/lts_vs_genetic'

# Iterate over files in directory, process the predictions and save them in a dict
# Print a table with aggregated results & create scatter plot (stat test if |values| > 1)