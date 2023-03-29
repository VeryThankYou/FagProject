import tensorflow.keras as ks
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv("submissions.csv")
upvotes = df["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
df["Logscore"] = logupvotes
ids = df["ID"]

def rename_ids():
    newids = []
    for e in ids:
        newids.append("resized_images/EarthPorn-" + e + ".png")
    return newids

names = rename_ids()
df["Filename"] = names

df1000 = df.iloc[:1000]

train_generator = ks.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2
    )

test_generator = ks.preprocessing.image.ImageDataGenerator(
    rescale = 1./255
    )

train_images = train_generator.flow_from_dataframe(
    dataframe=df1000,
    x_col='Filename',
    y_col='Score',
    target_size=(600, 600),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=df1000,
    x_col='Filename',
    y_col='Score',
    target_size=(600, 600),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=df1000,
    x_col='Filename',
    y_col='Score',
    target_size=(600, 600),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

inputs = ks.Input(shape = (600, 600, 3))
x = ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(inputs)
x = ks.layers.MaxPool2D()(x)
x = ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = ks.layers.MaxPool2D()(x)
x = ks.layers.GlobalAveragePooling2D()(x)
x = ks.layers.Dense(64, activation='relu')(x)
x = ks.layers.Dense(64, activation='relu')(x)
outputs = ks.layers.Dense(1, activation='linear')(x)

model = ks.Model(inputs = inputs, outputs = outputs)

model.compile(optimizer = "adam", loss = "mse")

history = model.fit(
    train_images,
    validation_data = val_images,
    epochs = 5,
    callbacks = [
        ks.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = 5,
            restore_best_weights = True
        )
    ]
)

predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels

rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("     Test RMSE: {:.5f}".format(rmse))

r2 = r2_score(true_ages, predicted_ages)
print("Test R^2 Score: {:.5f}".format(r2))
null_rmse = np.sqrt(np.sum((true_ages - np.mean(true_ages))**2) / len(true_ages))
print("Null/Baseline Model Test RMSE: {:.5f}".format(null_rmse))


print("W")