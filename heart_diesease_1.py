import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import regularizers


df = pd.read_csv("heart.csv").astype(int)
print(df.head())

print(df.dtypes)

for i in range(len(df.columns[:-1])):
    label = df.columns[i]
    for target in range(5):
        plt.hist(df[df["target"] == target][label], label=f"cancer {target}", alpha=0.7, density = True, bins =15)
    plt.title(label)
    plt.xlabel(label)
    plt.ylabel("N")
    plt.legend()
    plt.show()
df = df.dropna()
x = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

print('Class labels:', np.unique(y))
num_classes = len(np.unique(y))
print(num_classes)
    
scaler = StandardScaler()
x = scaler.fit_transform(x)
data = np.hstack((x,np.reshape(y, (-1, 1))))

x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size=0.5, random_state=0)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)


y_train = tf.keras.utils.to_categorical(y_train)
y_valid = tf.keras.utils.to_categorical(y_valid)
y_test = tf.keras.utils.to_categorical(y_test)



model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation= "relu", input_shape = (13,)),
    tf.keras.layers.Dense(16, activation = "relu"),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

model.fit(x_train, y_train, epochs = 60, batch_size = 15, validation_data =(x_valid, y_valid))

model.evaluate(x_test, y_test)

print("Train accuracy:", model.evaluate(x_train, y_train, verbose=0)[1])
print("Validation accuracy:", model.evaluate(x_valid, y_valid, verbose=0)[1])

y_pred = model.predict(x_test)

print(classification_report(y_test.argmax(axis=1), np.argmax(y_pred,axis=1)))


age = float(input("Enter age: "))
sex = float(input("Enter sex (0 for female, 1 for male): "))
cp = float(input("Enter chest pain type (0-3): "))
threstbps = float(input("Enter resting blood pressure: "))
chol = float(input("Enter serum cholesterol: "))
fbs = float(input("Enter fasting blood sugar > 120 mg/dl (1 for true, 0 for false): "))
restecg = float(input("Enter resting electrocardiographic results (0-2): "))
thalach = float(input("Enter maximum heart rate achieved: "))
exang = float(input("Enter exercise induced angina (1 for yes, 0 for no): "))
oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
slope = float(input("Enter the slope of the peak exercise ST segment (0-2): "))
ca = float(input("Enter number of major vessels colored by fluoroscopy (0-3): "))
thal = float(input("Enter thalassemia (3 for normal, 6 for fixed defect, 7 for reversable defect): "))


scaler = StandardScaler()
input_data = scaler.transform([[age, sex, cp, threstbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])


prediction = model.predict(input_data)
severity = np.argmax(prediction)


if severity == 0:
    print("Low risk of heart disease")
elif severity == 1:
    print("Moderate risk of heart disease")
elif severity == 2:
    print("High risk of heart disease")
elif severity == 3:
    print("Severe risk of heart disease")
elif severity == 4:
    print("Critical risk of heart disease")



