import numpy as np

# generer 15 integer compris entre -5  et 5
a = list(randint(-5, 6, 100) )

# 1000 experiments
# pick 200 samples with replacements
m = []
for i in range(1000):
    m.append(np.mean(random.choice(a, size = 200, replace = True)))

plt.boxplot(m)
print("np.mean(a) {:.4f} np.mean(m) {:.4f} +- {:.4f} ".format( np.mean(a), np.mean(m), 2* np.std(m) ))
print("95% Confidence  [{:.4f} {:.4f}]".format( np.percentile(a,10), np.percentile(a,90) ))
