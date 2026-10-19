import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from extract_and_inject import build_dashboard_data, inject_into_dashboard

#Reading the dataset and Analyzing it
data = pd.read_csv("creditcard.csv")
df = pd.DataFrame(data)
head = df.head()
info = df.info()
desc = df.describe()
print(f"Details Of the Raw Dataset \n{head}\n{info}\n{desc}")

# Creating A train-test split first to prevent data leakage
x = df.drop(columns=["Class"])
y = df["Class"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
#Creating Copies to suppress copy warning
x_train = x_train.copy()
x_test = x_test.copy()

#Time feature engineering
'''As the raw time column (shows elapsed seconds over the span of 48 hours) is not meaningful enough , we convert it into hierarchial features'''
x_train["Hour"] = (x_train["Time"] % 86400) // 3600
x_test["Hour"] = (x_test["Time"] % 86400) // 3600
x_train["Day"] = x_train["Time"] // 86400
x_test["Day"] = x_test["Time"] // 86400

x_train.drop(columns =["Time"],inplace = True)
x_test.drop(columns =["Time"],inplace = True)


#Performing IQR capping before feature scaling on amount column to remove robust outliers
Q1 = x_train["Amount"].quantile(0.25)
Q3 = x_train["Amount"].quantile(0.75)
IQR = Q3-Q1

lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR
'''Capping both train and test data using boundaries'''
x_train["Amount"] = x_train["Amount"].clip(lower_bound,upper_bound)
x_test["Amount"] = x_test["Amount"].clip(lower_bound,upper_bound)


#Feature scaling to remove outliers 
'''We are going to use standard scaling (z-score scaling)'''
scaler_amount = StandardScaler()
scaler_time = StandardScaler()
x_train[["Amount"]]= scaler_amount.fit_transform(x_train[["Amount"]])
x_test[["Amount"]] = scaler_amount.transform(x_test[["Amount"]])
#Hour and day are scaled together as both are time derived hierarchies
x_train[["Hour","Day"]] = scaler_time.fit_transform(x_train[["Hour","Day"]])
x_test[["Hour","Day"]] = scaler_time.transform(x_test[["Hour","Day"]]) 

# Shape check
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# No missing values
print(x_train.isnull().sum().sum())  # should print 0
print(x_test.isnull().sum().sum())   # should print 0

# Class distribution in y_test (should still reflect ~0.17% fraud)
print(y_test.value_counts(normalize=True))

#Percentages of values
print(y_train.value_counts(normalize=True))



'''Visualization For the CLeaned Data'''

#Class Distribution
plt.bar(y_train.value_counts().index,y_train.value_counts().values)
plt.xlabel("Class")
plt.ylabel("Transaction count")
plt.title("Class Distribution for Training set")
plt.show()

plt.bar(y_test.value_counts().index,y_test.value_counts().values)
plt.xlabel("Class")
plt.ylabel("Transaction count")
plt.title("Class Distribution for Testing set")
plt.show()

#Amount Distribution Before VS After 
plt.hist(df["Amount"],bins=50)
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Amount Distribution Before Scaling")
plt.show()
plt.hist(x_train["Amount"],bins=50)
plt.xlabel("Amount after Feature Scaling")
plt.ylabel("Frequency")
plt.show()

#Hour vs fraud rate
train_temp = pd.concat([x_train,y_train],axis=1)
fraud_only = train_temp[train_temp["Class"] == 1]
fraud_per_hour = fraud_only.groupby("Hour")["Class"].count()
plt.bar(fraud_per_hour.index, fraud_per_hour.values)
plt.xlabel("Hour")
plt.ylabel("Count of fraud transactions")
plt.title("Hour vs Fraud Rate")
plt.show()

#Day vs fraud rate
fraud_per_day = fraud_only.groupby("Day")["Class"].count()
plt.bar(fraud_per_day.index, fraud_per_day.values)
plt.xlabel("Day")
plt.ylabel("Count of fraud transactions")
plt.title("Day vs Fraud Rate")
plt.show()

#Correlation Heatmap
heat_data = x_train.corr()
sns.heatmap(heat_data,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


#Missing Values heatmap
miss_heat_data = x_train.isnull()
sns.heatmap(miss_heat_data,cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Fraud Probability Over Time
fraud_prob = train_temp.groupby("Hour")["Class"].mean()
plt.plot(fraud_prob.index, fraud_prob.values)
plt.xlabel("Hour")
plt.ylabel("Fraud Probability")
plt.title("Fraud Probability Over Time")
plt.show()

# Behavioral Differences (Fraud vs Normal)
normal_only = train_temp[train_temp["Class"] == 0]
features = ["Amount", "V4", "V11", "V14"]
for feature in features:
    plt.boxplot([normal_only[feature], fraud_only[feature]], labels=["Normal", "Fraud"])
    plt.title(f"Behavioral Difference — {feature}")
    plt.ylabel(feature)
    plt.show()

'''Model Building'''

#Random Forest Classification 
ran_forest = RandomForestClassifier(class_weight='balanced',random_state=42)
ran_forest.fit(x_train,y_train)
ran_forest_y_pred = ran_forest.predict(x_test)
random_forest_accuracy = accuracy_score(y_test,ran_forest_y_pred)
random_forest_precision = precision_score(y_test,ran_forest_y_pred)
random_forest_recall = recall_score(y_test,ran_forest_y_pred)
random_forest_f1 = f1_score(y_test,ran_forest_y_pred)
print(f"Accuracy of the Random Forest classifier is {random_forest_accuracy}")
print(f"Precision of the Random Forest classifier is {random_forest_precision}")
print(f"Recall score of the Random Forest classifier is {random_forest_recall}")
print(f"f1 score of the Random Forest classifier is {random_forest_f1}")
random_cm = confusion_matrix(y_test,ran_forest_y_pred)
random_cmdisp = ConfusionMatrixDisplay(confusion_matrix=random_cm,display_labels=["Normal","Fraud"])
random_cmdisp.plot()
plt.title("Confusion Matrix for Random forest Classifier")
plt.show()

#Logistic Regression
log_rec = LogisticRegression(class_weight='balanced',random_state=42,max_iter=1000) #max iter is 1000 as there are almost 30 PCA parameters (100 default is not enough)
log_rec.fit(x_train,y_train)
log_rec_y_pred = log_rec.predict(x_test)
log_rec_accuracy = accuracy_score(y_test,log_rec_y_pred)
log_rec_precision = precision_score(y_test,log_rec_y_pred)
log_rec_recall = recall_score(y_test,log_rec_y_pred)
log_rec_f1 = f1_score(y_test,log_rec_y_pred)
print(f"Accuracy of the Logistic Regression is {log_rec_accuracy}")
print(f"Precision of the Logistic Regression is {log_rec_precision}")
print(f"Recall score of the Logistic Regression is {log_rec_recall}")
print(f"f1 score of the Logistic Regression is {log_rec_f1}")
log_cm = confusion_matrix(y_test,log_rec_y_pred)
log_cmdisp = ConfusionMatrixDisplay(confusion_matrix=log_cm,display_labels=["Normal","Fraud"])
log_cmdisp.plot()
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

#Decision Tree Classifier
dec_tree = DecisionTreeClassifier(class_weight='balanced',random_state=42)
dec_tree.fit(x_train,y_train)
dec_tree_y_pred = dec_tree.predict(x_test)
dec_tree_accuracy = accuracy_score(y_test,dec_tree_y_pred)
dec_tree_precision = precision_score(y_test,dec_tree_y_pred)
dec_tree_recall = recall_score(y_test,dec_tree_y_pred)
dec_tree_f1 = f1_score(y_test,dec_tree_y_pred)
print(f"Accuracy of the Decision Tree Classifier is {dec_tree_accuracy}")
print(f"Precision of the Decision Tree Classifier is {dec_tree_precision}")
print(f"Recall score of the Decision Tree Classifier is {dec_tree_recall}")
print(f"f1 score of the Decision Tree Classifier is {dec_tree_f1}")
dec_cm = confusion_matrix(y_test,dec_tree_y_pred)
dec_cmdisp = ConfusionMatrixDisplay(confusion_matrix=dec_cm,display_labels=["Normal","Fraud"])
dec_cmdisp.plot()
plt.title("Confusion Matrix for Decision Tree Classifier")
plt.show()

feature_names = x_train.columns.tolist()   # lowercase x_train — matches your pipeline

data = build_dashboard_data(
    rf_model  = ran_forest,          rf_y_pred = ran_forest_y_pred,
    lr_model  = log_rec,             lr_y_pred = log_rec_y_pred,
    dt_model  = dec_tree,            dt_y_pred = dec_tree_y_pred,
    y_test    = y_test,
    x_test    = x_test,              # ← added for real fraud alerts
    feature_names = feature_names
)

inject_into_dashboard(data)
