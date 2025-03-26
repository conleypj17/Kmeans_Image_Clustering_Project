import numpy as np
from numpy import random as rand

rand.seed(5) #fix our random generation
data = rand.randint(0, 100, 100)
print(data)

#initialize centroid
centers = rand.randint(0, 100, k)
print(centers)

#use list to hold different groups
group0 = []
group1 = []
groups = [[]]

# define a stable condition
stable = False
threshold = 0.1 # represent the percentage of max centroid change

#start k-means repeating process

iteration = 0
while not stable and iteration < 100:
    iteration = iteration + 1
    #traverse each number and calculate its distance to each centroid
    #pick the smaller one as group assignment
    for i in range (data.shape[0]):
        distance0 = abs(data[i] - centers[0]) #compute distance to the first centroid
        distance1 = abs(data[i] - centers[1]) #compute distance to the second centroid
        if distance0 < distance1:
            group0.append(data[i])
        else:
            group1.append(data[i])

    #re-compute the new centroid
    old_centers = centers
    centers[0] = sum(group0) / len(group0)
    centers[1] = sum(group1) / len(group1)

    group0_np = np.array(group0)
    group1_np = np.array(group1)
    print(group0_np)
    print(group1_np)

    #check if the centers get stable
    if abs(centers[0] - old_centers[0]) < threshold and abs(centers[1] - old_centers[1]) < threshold:
        stable = True
