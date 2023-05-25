import numpy as np
#np.object_ = object
import PIL
import transformers as tf
import torch
from PIL import Image
import requests
import sklearn.metrics as skm
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = skm.mean_squared_error(labels, logits)
    rmse = skm.mean_squared_error(labels, logits, squared=False)
    mae = skm.mean_absolute_error(labels, logits)
    r2 = skm.r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}



#impipeline = pipeline(model = "facebook/detr-resnet-50")
#model = impipeline.model
image = PIL.Image.open("resized_images/EarthPorn-1a3x6n.png")

processor = tf.AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = tf.ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels = 1, ignore_mismatched_sizes = True).to("cuda")

"""""
training_args = tf.TrainingArguments(
    output_dir ='./results',          
    num_train_epochs = num_epochs,     
    per_device_train_batch_size = 64,   
    per_device_eval_batch_size = 20,   
    weight_decay = 0.01,               
    learning_rate = 2e-5,
    logging_dir = './logs',            
    save_total_limit = 10,
    load_best_model_at_end = True,     
    metric_for_best_model = 'rmse',    
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
) 

# Call the Trainer
trainer = tf.Trainer(
    model = model,                         
    args = training_args,                  
    train_dataset = train_dataset,         
    eval_dataset = valid_dataset,          
    compute_metrics = compute_metrics_for_regression,     
)
"""""
#inputs = processor(images=image, return_tensors="pt")
#outputs = model(**inputs)
#print(outputs)