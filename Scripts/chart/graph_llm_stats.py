import matplotlib.pyplot as plt
import numpy as np

# Script to visualize edit distances between LLM responses
# Author: Christopher Rauch
# Date: 2025

# Define the models and categories
models = ['Claude 3.7 Sonnet', 'GPT-4o', 'Gemini 2.0 flash', 'Gemini 1.5-pro', 'GPT-4.1', 'GPT-o3']
categories = ['Car', 'Body', 'Computer', 'Job']

# Initialize data structures to store results
averages = {model: [] for model in models}
std_devs = {model: [] for model in models}
minimums = {model: [] for model in models}
maximums = {model: [] for model in models}

# Hard-coded data from the paper
paper_data = {
    'Claude 3.7 Sonnet': {
        'Car': {'avg': 0.17, 'std': 0.01, 'min': 0.14, 'max': 0.20},
        'Body': {'avg': 0.17, 'std': 0.01, 'min': 0.15, 'max': 0.20},
        'Computer': {'avg': 0.17, 'std': 0.01, 'min': 0.14, 'max': 0.23},
        'Job': {'avg': 0.16, 'std': 0.01, 'min': 0.14, 'max': 0.21}
    },
    'GPT-4o': {
        'Car': {'avg': 0.17, 'std': 0.01, 'min': 0.14, 'max': 0.21},
        'Body': {'avg': 0.17, 'std': 0.01, 'min': 0.15, 'max': 0.21},
        'Computer': {'avg': 0.17, 'std': 0.01, 'min': 0.14, 'max': 0.20},
        'Job': {'avg': 0.17, 'std': 0.01, 'min': 0.15, 'max': 0.19}
    },
    'Gemini 2.0 flash': {
        'Car': {'avg': 0.20, 'std': 0.01, 'min': 0.19, 'max': 0.23},
        'Body': {'avg': 0.20, 'std': 0.01, 'min': 0.18, 'max': 0.23},
        'Computer': {'avg': 0.20, 'std': 0.01, 'min': 0.18, 'max': 0.22},
        'Job': {'avg': 0.19, 'std': 0.01, 'min': 0.17, 'max': 0.21}
    },
    'Gemini 1.5-pro': {
        'Car': {'avg': 0.21, 'std': 0.01, 'min': 0.19, 'max': 0.24},
        'Body': {'avg': 0.20, 'std': 0.01, 'min': 0.18, 'max': 0.22},
        'Computer': {'avg': 0.19, 'std': 0.01, 'min': 0.17, 'max': 0.21},
        'Job': {'avg': 0.19, 'std': 0.01, 'min': 0.17, 'max': 0.21}
    },
    'GPT-4.1': {
        'Car': {'avg': 0.22, 'std': 0.01, 'min': 0.20, 'max': 0.24},
        'Body': {'avg': 0.23, 'std': 0.01, 'min': 0.21, 'max': 0.25},
        'Computer': {'avg': 0.21, 'std': 0.01, 'min': 0.19, 'max': 0.23},
        'Job': {'avg': 0.20, 'std': 0.01, 'min': 0.18, 'max': 0.22}
    },
    'GPT-o3': {
        'Car': {'avg': 0.19, 'std': 0.02, 'min': 0.16, 'max': 0.28},
        'Body': {'avg': 0.19, 'std': 0.02, 'min': 0.15, 'max': 0.26},
        'Computer': {'avg': 0.18, 'std': 0.02, 'min': 0.14, 'max': 0.27},
        'Job': {'avg': 0.17, 'std': 0.02, 'min': 0.13, 'max': 0.26}
    }
}

# Extract data from the hard-coded values into our data structures
for model in models:
    for category in categories:
        data = paper_data[model][category]
        averages[model].append(data['avg'])
        std_devs[model].append(data['std'])
        minimums[model].append(data['min'])
        maximums[model].append(data['max'])

# Create a color scheme for the bars - one color per model
# Color associations by model type
# Claude: blue
# GPT models: shades of red/orange
# Gemini models: shades of green
colors = {
    'Claude 3.7 Sonnet': 'royalblue',
    'GPT-4o': 'firebrick',
    'Gemini 2.0 flash': 'seagreen',
    'Gemini 1.5-pro': 'mediumseagreen',
    'GPT-4.1': 'darkorange',
    'GPT-o3': 'indianred'
}

# List of colors in the same order as models for direct indexing
color_list = [colors[model] for model in models]

# Set up bar positions
x = np.arange(len(categories))
width = 0.14  # Width of bars

# Create figure and plot with taller height to accommodate legends
fig, ax = plt.subplots(figsize=(16, 12))

# Create bars for each model
for i, model in enumerate(models):
    position = x + (i - 2.5) * width 
    
    # Plot the average bars with error bars for standard deviation
    bars = ax.bar(position, averages[model], width, label=model, color=color_list[i], 
                 yerr=std_devs[model], capsize=5)
    
    # Add min-max lines
    for j, (bar, min_val, max_val) in enumerate(zip(bars, minimums[model], maximums[model])):
        bar_x = bar.get_x() + bar.get_width()/2
        ax.plot([bar_x, bar_x], [min_val, max_val], color='black', linewidth=1.5)
        
        # Add small horizontal lines at min and max
        min_line_width = width/3
        ax.plot([bar_x - min_line_width/2, bar_x + min_line_width/2], [min_val, min_val], color='black', linewidth=1.5)
        ax.plot([bar_x - min_line_width/2, bar_x + min_line_width/2], [max_val, max_val], color='black', linewidth=1.5)
        
        # Add text annotations for the average value with background
        ax.text(bar_x, bar.get_height() + 0.020, f'{averages[model][j]:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1, boxstyle='round'))

# Add a horizontal line at y=0.2 for reference
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)

# Add labels, title, and legend
ax.set_xlabel('Problem Domain', fontweight='bold', fontsize=12)
ax.set_ylabel('Edit Distance', fontweight='bold', fontsize=12)
ax.set_title('Edit Distance between LLM Responses to First and Second Prompts', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight='bold')

# Set y-axis limits with some padding to accommodate the maximum values
y_max = max([max(maximums[model]) for model in models])
ax.set_ylim(0, y_max + 0.1)

# Add custom legend for statistics
custom_legend = [
    plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label='Min-Max Range'),
    plt.errorbar([], [], yerr=1, fmt='none', color='black', capsize=5, label='Avg ± Std Dev')
]

# Add grid lines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Create colored handles for model legend with larger line width for better visibility
model_handles = []
for i, model in enumerate(models):
    model_handles.append(plt.Line2D([0], [0], color=color_list[i], lw=6, label=model))

# Position legends with clear spacing
# Create the model legend with colored text - larger and more prominent
model_legend = plt.legend(handles=model_handles, loc='upper center', 
                        bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=12,
                        title="LLM Models", title_fontsize=14)
plt.gca().add_artist(model_legend)  # Add the model legend

# Add statistics legend in a different position
stats_legend = plt.legend(handles=custom_legend, loc='upper right', fontsize=10)
plt.gca().add_artist(stats_legend)  # Add the statistics legend

# Save the figure with high resolution (PNG only)
# Don't use tight_layout() as it can sometimes conflict with manual positioning
plt.savefig('llm_edit_distance_comparison.png', dpi=300)

print("Graph generated and saved as 'llm_edit_distance_comparison.png'")
plt.close()  # Close the figure to avoid displaying it
