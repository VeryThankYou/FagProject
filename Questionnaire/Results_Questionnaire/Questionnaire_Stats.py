import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import os 

# Set directory
os.chdir(os.getcwd()+"/Questionnaire/Results_Questionnaire")

# Create list of dictionaries to convert the scrambled order of images to the real order
txtfile = open("order.txt","r")
lines = txtfile.readlines()
lines = [e[:-1] for e in lines[1:]]
# To read, index by question number - 1, then index image number - 1. Then you get the model number.
# 0 = pretrained model, 1 = low-upvote model, 2 = all data model, 3 = high-upvote model
translator = [{int(i): int(e) for i, e in enumerate(line.split(":"))} for line in lines]

# Read the responses, sort so only the respondants who consented are shown
df = pd.read_csv("FormResponses.csv")
df = df[df.iloc[:, 1] == "I consent"]

# Create np array of answers
# Every row is a single respondent's answers to a single question
# First column is their favourite image of the question, second is their second-favorite and so on
stackedResponses = np.zeros((df.shape[0] * 16, 4))
for i in range(16):
    dfQ = df.iloc[:, (2 + i * 4):(2 + (i + 1) * 4)]
    for i2 in range(4):
        column = dfQ.iloc[:, i2]
        col2 = [int(e[-1]) for e in column]
        stackedResponses[i * df.shape[0]:(i + 1) * df.shape[0], i2] = col2
print(stackedResponses)
print(stackedResponses.shape)

# Hypothesis A:
# All data is better than pretrained
hyp1 = 0

# Hypothesis B:
# Bad data is worse than pretrained
hyp2 = 0

# Hypothesis C:
# Good data is better than pretrained
hyp3 = 0

# Hypothesis D:
# Good data is better than all data
hyp4 = 0

# Hypothesis E:
# Good data is better than bad data
hyp5 = 0

# Hypothesis F
bestIsBest = 0

# Kinda hypothesis
worstPreIsWorst = 0

# Count every occurance that supports a hypothesis
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
    if votes[0] == 3 or votes[1] == 3:
        worstPreIsWorst = worstPreIsWorst + 1

# Define number of data points
nobs = stackedResponses.shape[0]

# Print proportions
print(hyp1/nobs)
print(hyp2/nobs)
print(hyp3/nobs)
print(hyp4/nobs)
print(hyp5/nobs)
print(bestIsBest/nobs)

# Compute z-tests, hypothesis F is done singularly, as the null-hypothesis-proportion is different
value = 0.5
hypcounts = [hyp1, hyp2, hyp3, hyp4, hyp5]
table = {"Proportions": [hyp/nobs for hyp in hypcounts]}
table["Proportions"].append(bestIsBest/nobs)
statpval = [proportions_ztest(hyp, nobs, value, alternative = "larger", prop_var=value) for hyp in hypcounts]
table["Test statistic"] = [e[0] for e in statpval]
table["P-value"] = [e[1] for e in statpval]

stat, pval = proportions_ztest(bestIsBest, nobs, 0.25, alternative = "larger", prop_var=0.25)
table["P-value"].append(pval)
pvals = [(e[0] + 1, e[1]) for e in enumerate(table["P-value"])]
table["Test statistic"].append(stat)
pvalssorted = sorted(pvals, key=lambda x: x[1])
adjustVals = [(e[0], e[1] * (len(hypcounts) + 1) / (i+1)) for i, e in enumerate(pvalssorted)]
table["Adjusted P-value"] = [e[1] for e in sorted(adjustVals, key=lambda x: x[0])]
print(adjustVals)

# Build and print table of results
table = pd.DataFrame(data=table)
print(table)







stat, pval = proportions_ztest(worstPreIsWorst, nobs, 0.5, alternative = "larger", prop_var=0.5)
print(pval)