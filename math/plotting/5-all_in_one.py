#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 1st plot
y0 = np.arange(0, 11) ** 3

# 2nd plot
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

# 3rd plot
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

# 4th plot
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

# 5th plot
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create figure and axis
#fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))

fig = plt.figure(constrained_layout=True)
grid_spec = fig.add_gridspec(3, 2)
fig.suptitle("All in one")
# Create 1st plot
plt0 = fig.add_subplot(grid_spec[0:1])
plt0.plot(y0, color="red")
plt0.set_xlim(left=0, right=10)
plt0.set_ylim(bottom=0, top=1000)

# Create 2nd plot
plt1 = fig.add_subplot(grid_spec[1:2])
plt1.scatter(x1, y1, color="magenta")
plt1.set_title("Men's Height vs Weight", fontsize="x-small")
plt1.set_ylabel("Weight (lbs)", fontsize="x-small")
plt1.set_xlabel("Height (in)", fontsize="x-small")

# Create 3rd plot
plt2 = fig.add_subplot(grid_spec[2:3])
plt2.plot(x2, y2, color="blue")
plt2.set_title("Exponential Decay of C-14", fontsize="x-small")
plt2.set_ylabel("Fraction Remaining", fontsize="x-small")
plt2.set_xlabel("Time (years)", fontsize="x-small")
plt2.set_yscale("log")
plt2.set_xticks([0, 10000, 20000])
plt2.set_xlim(left=0, right=30000)

# Create 4th plot
plt3 = fig.add_subplot(grid_spec[3:4])
plt3.plot(x3, y31, linestyle="dashed", color="red", label="C-14")
plt3.plot(x3, y32, color="green", label="Ra-226")
plt3.set_title("Exponential Decay of Radioactive Elements", fontsize="x-small")
plt3.set_xlim(left=0, right=20000)
plt3.set_xticks([0, 5000, 10000, 15000, 20000])
plt3.set_yticks([0, 0.5, 1])
plt3.set_ylabel("Fraction Remaining", fontsize="x-small")
plt3.set_xlabel("Time (years)", fontsize="x-small")

# Create 5th plot
plt4 = fig.add_subplot(grid_spec[4:6])
plt4.hist(student_grades, bins=10, color="blue", edgecolor="black")
plt4.set_title("Project A", fontsize="x-small")
plt4.set_xlabel("Grades", fontsize="x-small")
plt4.set_ylabel("Frequency", fontsize="x-small")
plt4.set_xlim(left=0, right=100)

# Show plot
plt.show()