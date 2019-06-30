import numpy as np

import matplotlib.pyplot as plt

ys = []

with open('loss.log', 'r') as f:
    ys = [float(s) for s in f.read().split()]

xs = [i+10 for i in range(0, 10*len(ys), 10)]
print (xs)
print (ys)

plt.plot(xs, ys)
plt.ylabel('Number of Samples')
plt.xlabel('Loss')
plt.title('Loss plot with batch size of 10')
plt.savefig('docs/loss.png', dpi=300)
plt.show()

