import matplotlib.pyplot as plt

# Example data
crops = ['Rice', 'Wheat', 'Corn', 'Barley']
scores = [85, 78, 92, 70]

plt.bar(crops, scores, color=['green', 'blue', 'red', 'purple'])
plt.xlabel('Crops')
plt.ylabel('Compatibility Score')
plt.title('Crop Compatibility with Given Conditions')
plt.ylim(0, 100)  # Score out of 100
plt.show()
