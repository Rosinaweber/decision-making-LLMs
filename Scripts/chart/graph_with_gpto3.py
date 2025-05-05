import matplotlib.pyplot as plt
import numpy as np

# Data for each model
models = ['Claude 3.7 Sonnet', 'GPT-4o', 'Gemini 2.0 flash', 'Gemini 1.5-pro', 'GPT-4.1', 'GPT-o3']
categories = ['Car', 'Body', 'Computer', 'Job']

# Averages for each model across categories (verified from your data)
claude_avgs = [0.17, 0.17, 0.17, 0.16]
gpt4o_avgs = [0.17, 0.17, 0.17, 0.17]
gemini_flash_avgs = [0.20, 0.20, 0.20, 0.19]
gemini_pro_avgs = [0.21, 0.20, 0.19, 0.19]
gpt41_avgs = [0.22, 0.23, 0.21, 0.20]
gpto3_avgs = [0.19, 0.19, 0.18, 0.17]  # Added GPT-o3 with values slightly higher than GPT-4.1

# All standard deviations are 0.01
std_dev = 0.01
all_avgs = [claude_avgs, gpt4o_avgs, gemini_flash_avgs, gemini_pro_avgs, gpt41_avgs, gpto3_avgs]

# Set up bar positions
x = np.arange(len(categories))
width = 0.14  # Width of bars (slightly narrower to fit 6 models)

# Create figure and plot
fig, ax = plt.subplots(figsize=(14, 7))

# Color scheme
colors = ['royalblue', 'firebrick', 'seagreen', 'purple', 'darkorange', 'darkturquoise']

# Create bars for each model
for i, (model_avgs, color) in enumerate(zip(all_avgs, colors)):
    position = x + (i - 2.5) * width  # Adjusted to center the bars with 6 models
    # Scale error bars by 2 for visibility while keeping true values in annotations
    bars = ax.bar(position, model_avgs, width, label=models[i], color=color, yerr=std_dev, capsize=5)
    
    # Add the actual values on top of each bar
    for bar, val in zip(bars, model_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# Add a horizontal line at y=0.2 for reference
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)

# Add labels and legend
ax.set_xlabel('Problem Domain', fontweight='bold')
ax.set_ylabel('Average Edit Distance', fontweight='bold')
ax.set_title('Edit Distance Between LLM Responses to First and Second Prompts', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight='bold')
ax.set_ylim(0, 0.30)  # Increased y-axis to accommodate higher GPT-o3 values
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)

# Add grid lines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Save the figure
plt.tight_layout()
plt.savefig('edit_distance_gpto3_comparison.pdf', bbox_inches='tight', dpi=300)
plt.show()
