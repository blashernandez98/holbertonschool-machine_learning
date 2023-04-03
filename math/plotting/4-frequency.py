#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Define seed and generate random collection
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Define the bin edges for the histogram
bins = np.arange(0, 110, 10)

# Create the histogram
hist, bin_edges = np.histogram(student_grades, bins=bins)

# Create the bar plot
plt.bar(bin_edges[:-1], hist, width=10, align="edge", edgecolor="black")
plt.xticks(bin_edges)

# Set the title and labels for the X-axis and Y-axis
plt.title("Project A")
plt.xlabel('Grades')
plt.ylabel('Number of students')
plt.xlim(0, 100)
plt.ylim(0, 30)

# Show the plot
plt.show()