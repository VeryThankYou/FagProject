import numpy as np
import pandas as pd
import transformers as tf
import torch
from PIL import Image
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import utils


class CustomImageDataset(Dataset):
    def __init__(self, Xs, ys):
        self.img_labels = ys
        self.inputs = Xs


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = {"pixel_values": torch.as_tensor(self.inputs[idx]["pixel_values"][0]).float()}
        image["labels"] = torch.as_tensor(self.img_labels[idx]).float()
        return image

processor = tf.AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = tf.ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels = 1, ignore_mismatched_sizes = True).to("cuda")


df = pd.read_csv("submissions86000.csv")
upvotes = df["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
df["Logscore"] = logupvotes
ids = df["ID"]



def rename_ids():
    newids = []
    for e in ids:
        newids.append("all_data256/EarthPorn-" + e + ".png")
    return newids

names = rename_ids()
df["Filename"] = names
df1000 = df.iloc[:10000]
X = [processor(Image.open(e)) for e in df["Filename"]]

y = df["Logscore"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = skm.mean_squared_error(labels, logits)
    rmse = skm.mean_squared_error(labels, logits, squared=False)
    mae = skm.mean_absolute_error(labels, logits)
    r2 = skm.r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}




training_args = tf.TrainingArguments(
    output_dir ='./results86',          
    num_train_epochs = 100,     
    per_device_train_batch_size = 24,   
    per_device_eval_batch_size = 20,   
    weight_decay = 0.01,               
    learning_rate = 2e-5,
    logging_dir = './logs86',            
    save_total_limit = 10,
    load_best_model_at_end = True,     
    metric_for_best_model = 'rmse',    
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
) 

train_dataset = CustomImageDataset(X_train, y_train)
valid_dataset = CustomImageDataset(X_test, y_test)



# Call the Trainer
trainer = tf.Trainer(
    model = model,                         
    args = training_args,                  
    train_dataset = train_dataset,         
    eval_dataset = valid_dataset,          
    compute_metrics = compute_metrics_for_regression,     
)

trainer.train()
trainer.evaluate()


