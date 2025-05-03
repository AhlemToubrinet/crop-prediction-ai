import pip
pip.main(["install","matplotlib"])

from matplotlib import pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a basic line graph
plt.plot(x, y)

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Simple Line Graph')

# Display the graph
plt.show()