import pandas as pd
import numpy as np

txtfile = open("order.txt","r")
lines = txtfile.readlines()

lines = [e[:-1] for e in lines[1:]]
translator = [{int(i): int(e) for i, e in enumerate(line.split(":"))} for line in lines]


df = pd.read_csv("FormResponses.csv")
df = df[df["Consent"] == "I consent"]
print(df)