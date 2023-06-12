import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

txtfile = open("order.txt","r")
lines = txtfile.readlines()

lines = [e[:-1] for e in lines[1:]]
translator = [{int(i): int(e) for i, e in enumerate(line.split(":"))} for line in lines]


df = pd.read_csv("FormResponses.csv")
df = df[df["Consent"] == "I consent"]
stackedResponses = np.zeros((df.shape[0] * 16, 4))
for i in range(16):
    indices = ["Q" + str(i + 1) + "Best" + str(i2 + 1) for i2 in range(4)]
    dfQ = df[indices]
    for i2 in range(4):
        column = dfQ.iloc[:, i2]
        col2 = [int(e[-1]) for e in column]
        stackedResponses[i * df.shape[0]:(i + 1) * df.shape[0], i2] = col2
print(stackedResponses)
print(stackedResponses.shape)

# Hypothesis 1:
# All data is better than pretrained
hyp1 = 0

# Hypothesis 2:
# Bad data is worse than pretrained
hyp2 = 0

# Hypothesis 3:
# Good data is better than pretrained
hyp3 = 0

# Hypothesis 4:
# Good data is better than all data
hyp4 = 0

# Hypothesis 5:
# Good data is better than bad data
hyp5 = 0

bestIsBest = 0

for i in range(stackedResponses.shape[0]):
    question = int(i / df.shape[0])
    #print(stackedResponses[i, :], question)
    votes = {translator[question][int(e) - 1]: i2 for i2, e in enumerate(stackedResponses[i, :])}
    #print(votes)
    if votes[2] < votes[0]:
        hyp1 = hyp1 + 1
    if votes[1] > votes[0]:
        hyp2 = hyp2 + 1
    if votes[3] < votes[0]:
        hyp3 = hyp3 + 1
    if votes[3] < votes[2]:
        hyp4 = hyp4 + 1
    if votes[3] < votes[1]:
        hyp5 = hyp5 + 1
    if votes[3] == 0:
        bestIsBest = bestIsBest + 1

nobs = stackedResponses.shape[0]
value = 0.5

print(stackedResponses.shape)
print(hyp1/nobs)
print(hyp2/nobs)
print(hyp3/nobs)
print(hyp4/nobs)
print(hyp5/nobs)
print(bestIsBest/nobs)
hypcounts = [hyp1, hyp2, hyp3, hyp4, hyp5]
table = {"Proportions": [hyp/nobs for hyp in hypcounts]}
statpval = [proportions_ztest(hyp, nobs, value, alternative = "larger") for hyp in hypcounts]
table["Test statistic"] = [e[0] for e in statpval]
table["P-value"] = [e[1] for e in statpval]
table = pd.DataFrame(data=table)
print(table)



stat, pval = proportions_ztest(hyp5, nobs, value, alternative = "larger")
print(pval)


stat, pval = proportions_ztest(bestIsBest, nobs, 0.25, alternative = "larger")
print(pval)