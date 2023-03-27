import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import regularizers



cols = ["age", "sex", "cp", "threstbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
cleveland = pd.read_csv("processed.cleveland.data", sep=",", names=cols, encoding="utf-8-sig")
switzerland = pd.read_csv("processed.switzerland.data", sep=",", names=cols, encoding="utf-8-sig")
hungary = pd.read_csv("reprocessed.hungarian.data", sep=" ", names=cols, encoding="utf-8-sig")
#longbeach = pd.read_csv("processed.va.data", sep=",", encoding="utf-8-sig")


merged1 = pd.concat([cleveland, switzerland], ignore_index=True)
df = pd.concat([merged1, hungary], ignore_index= True )

#df = pd.concat([merged2, longbeach], ignore_index= True)

#df = df.fillna()


df["threstbps"] = pd.to_numeric(df["threstbps"], errors='coerce')
df["fbs"] = pd.to_numeric(df["fbs"], errors='coerce')
df["restecg"] = pd.to_numeric(df["restecg"], errors='coerce')
df["thalach"] = pd.to_numeric(df["thalach"], errors='coerce')
df["exang"] = pd.to_numeric(df["exang"], errors='coerce')
df["oldpeak"] = pd.to_numeric(df["oldpeak"], errors='coerce')
df["slope"] = pd.to_numeric(df["slope"], errors='coerce')
df["ca"] = pd.to_numeric(df["ca"], errors='coerce')
df["thal"] = pd.to_numeric(df["thal"], errors='coerce')
df["num"] = pd.to_numeric(df["num"], errors="coerce")

#for i in range(len(df.columns[:-1])):
    label = df.columns[i]
    for num in range(5):
        plt.hist(df[df["num"] == num][label], label=f"cancer {num}", alpha=0.7, density = True, bins =15)
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

#over = RandomOverSampler()
#x,y = over.fit_resample(x, y)
#data = np.hstack((x, np.reshape(y, (-1, 1))))
#transformed_df = pd.DataFrame(data, columns=df.columns)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.1, random_state=0)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)


y_train = tf.keras.utils.to_categorical(y_train)
y_valid = tf.keras.utils.to_categorical(y_valid)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation= "relu", input_shape = (13,), kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dense(16, activation = "relu", kernel_regularizer=regularizers.l1(0.001)),
    #tf.keras.layers.Dense(16, activation = "relu", kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dense(5, activation='softmax')
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
else:
    print("Critical risk of heart disease")



