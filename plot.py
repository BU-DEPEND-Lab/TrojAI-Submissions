import numpy as np
import matplotlib.pyplot as plt

# Data
x = [-2.2441144, -0.12808709, -4.2543817]
y = [-3.5665362, -3.2295027, -0.07024086]
labels = ['Item 1', 'Item 2', 'Item 3']  # Optional labels for the bars

# Set the positions for the bars
x_positions = np.arange(len(x))

# Set the width of each pair of bars
bar_width = 0.4

# Create the bar chart
plt.bar(x_positions - bar_width / 2, x, width=bar_width, label='X Data', color='blue', alpha=0.7)
plt.bar(x_positions + bar_width / 2, y, width=bar_width, label='Y Data', color='green', alpha=0.7)

# Add labels and a legend
plt.xlabel('Items')
plt.ylabel('Values')
plt.xticks(x_positions, labels)
plt.legend()

# Show the chart
plt.savefig('bar_chart.png')