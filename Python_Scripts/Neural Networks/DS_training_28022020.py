# Data Sci training
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('C:\\Users\\sakkarkr\\Downloads\\Week4_2 Imbalance Data - Student\\Data\\VehChoice.csv')
dataset.set_index('ID', inplace = True)

dataset.loc[dataset['Vehicle'] == 'C', 'Vehicle'] = 1
dataset.loc[dataset['Vehicle'] == 'M', 'Vehicle'] = 0

cat_var = ['Gender','Driving Area','Vehicle Use']

fig, axs = plt.subplots(nrows = len(cat_var), figsize= (8, 20), sharex = False)
for i in range(len(cat_var)):
    sns.countplot(x = cat_var[i], data = dataset, hue = 'Vehicle', ax = axs[i])


# LabelEncoder()
le_sex = LabelEncoder()
sex = le_sex.fit_transform(dataset['Gender'])

le_DrivingArea = LabelEncoder()
DrivingArea = le_DrivingArea.fit_transform(dataset['Driving Area'])

le_VehicleUse = LabelEncoder()
VehicleUse = le_VehicleUse.fit_transform(dataset['Vehicle Use'])

oh_all = OneHotEncoder(sparse = False, dtype = int)
all_onehot = oh_all.fit_transform(np.stack([sex, DrivingArea, VehicleUse], axis = 1))


columns=['Sex_'+str(c) for c in le_sex.classes_]
columns+=['DrivingArea_'+str(c) for c in le_DrivingArea.classes_]
columns+=['VehicleUse_'+str(c) for c in le_VehicleUse.classes_]

onehot_df = pd.DataFrame(all_onehot, index=dataset.index, columns=columns)
onehot_df = onehot_df.drop(columns=['Sex_Female','DrivingArea_Rural','VehicleUse_Commercial'], axis=1)
onehot_df.head()

data = pd.concat([dataset, onehot_df], axis = 1)

# Binary

bins = [0,30,40,50,60,data['Age'].max()]
data_age_binned = pd.cut(data['Age'], bins)
print('\\nNumber of customers:\\n{}'.format(data['Age'].groupby(data_age_binned).count()))

le_age_binned = LabelEncoder()
age_binned = le_age_binned.fit_transform(data_age_binned)
data = data.assign(Age_bin = age_binned)

# Split train/test
select_data = data[['Sex_Male', 'DrivingArea_Urban', 'VehicleUse_Private', 'Age_bin', 'Price.C', 'Price.M', 'Price.F', 'Vehicle']]
train, test = train_test_split(select_data, test_size = 0.3 , random_state = 42)

features = ['Sex_Male', 'DrivingArea_Urban', 'VehicleUse_Private', 'Age_bin', 'Price.C']
target = ['Vehicle']

X_train = train[features]
y_train = train[target]
X_val = test[features]
y_val = test[target]

model_logreg = LogisticRegression()
model_logreg.fit(X_train, y_train)

intercept = 0
coef = 0
coef = model_logreg.coef_
list(zip(features, coef[0]))

# Predict
y_train_pred_class = model_logreg.predict(X_train)
y_val_pred_class = model_logreg.predict(X_val)

# Evalution
acc_score_y_train_pred = accuracy_score(y_train, y_train_pred_class)
acc_score_y_val_pred = accuracy_score(y_val, y_val_pred_class)
class_report = classification_report(y_val, y_val_pred_class)
conf_matrix = confusion_matrix(y_val, y_val_pred_class)
 
print('Train Accuracy :',acc_score_y_train_pred)
print('Validation Accuracy :',acc_score_y_val_pred,'\n')
print('Classification Report','\n')
print(class_report,'\n')
print('Confusion Matrix','\n')
print(conf_matrix)


# Threshold

# Resampling 
df_majority = train[train['Vehicle'] == 1]
df_minority = train[train['Vehicle'] == 0]

# Downsample
df_majority_downsampled = resample(df_majority,
                                   replace = False,
                                   n_samples = df_minority.shape[0],
                                   randam_state = 42)
# combine minority
train_downsampled = pd.concat([df_majority_downsampled, df_minority])

features = ['Sex_Male', 'DrivingArea_Urban', 'VehicleUse_Private', 'Age_bin', 'Price.C']
target = ['Vehicle']

X_train = train_downsampled[features]
y_train = train_downsampled[target]

X_val = test[features]
y_val = test[target]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("\n")
print("Class 1:", train_downsampled[train_downsampled['Vehicle']==1].shape)
print("Class 0:", train_downsampled[train_downsampled['Vehicle']==0].shape)


y_train_pred_class = model_logreg.predict_proba(X_train)
y_val_pred_class = model_logreg.predict_proba(X_val)

row = []
for i in range(len(y_val_pred_class.tolist())):
    row.append([y_val.values[i][0], y_val_pred_class[i][0],y_val_pred_class[i][1]])

results = pd.DataFrame(row, columns = ['Actual','Class0','Class1'])

results.head()









