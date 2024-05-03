# <p align="center"> Bank-Loan-Defaulter-Prediction-Project-Using-Python </p>
 
# <p align="center">![images](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/ce7ec7b3-01be-42f1-836b-0f5d6828b7c5)


</p>

### Introduction

Loan default prediction is a crucial task for banks and financial institutions to mitigate risks associated with lending.This project utilizes machine learning algorithms to analyze 
customer data and predict whether a customer is likely to default on their loan.

### Dependencies

To run this project, you need the following dependencies:

- python3
- pandas
- numpy
- sk learn
- seaborn
- matplotlib
- jupyter lab
- Excel

[Dataset Used](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/blob/36cace2def289bf8d78cad40bafe89359988c61a/bankloans%20data.xls)

[Python Script](Bank_loans_script.ipynb)

- ### Information about our dataset

```py
data_ex.info()
```

###### result

![Img_1](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/467fc5b2-45c3-44c9-affe-494efcd76417)

 - ##### We are not having any null values.

```py
#2.missing values treatment
data_ex.isnull().sum()
```

###### Result

![Img_2](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/c7f0e775-5015-4147-ad7a-b5f7874862de)

- ### Outlier treatment 

##### We have used box plot for Outlier detection.

```py
#3.outliar treatment 

for column in data_ex:
    plt.figure(figsize=(10,1))
    sns.boxplot(x=data_ex[column])
    plt.title(f"Boxplot of {column}")
    plt.show() 
```
###### Result 

![Img_3](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/0667ca86-efe3-418a-8441-d8c3e60b9dc4)

![Img_4](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/0b421cc0-94d2-458b-bf34-a967e23bd8d0)

![Img_5](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/59263b2f-51ae-4eb2-ac52-ca5984a4186e)

- #### checking the distribution of the data

```py
#4.checking the distribution of the data
#Using random distribution 
for col in d1.columns:
    sns.displot(d1[col],kde = True)
```

###### Result:

![Img_6](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/16113edf-0830-483e-9319-e1241058f20e)
![Img_7](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/9cb6adb8-3f63-4e4a-98cf-029ba5448ceb)

![Img_8](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/c5b702da-50d8-4bda-bb93-7ba412244ea6)
![Img_9](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/ae87d3d0-333f-44ed-9573-c647e73c922b)

![Img_10](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/4fcf5897-3ef0-423b-aad3-f80a1cc0a0a0)
![Img_11](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/841ad88a-f04c-42b5-9d34-305eab599a30)

![Img_12](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/75f3b677-65f9-44e8-9456-9f6cf0b60162)
![Img_13](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/d3f6256d-5911-4e7d-ab5f-43363d9a7019)

- #### Correlation 
```py
#6.correlation analysis 
d1.corr()
sns.heatmap(d1.corr().abs(),annot=True)
plt.show()
```
![Img_14](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/5a0f9d11-97b8-406d-b937-793a51b68343)

- ### Model Evaluation
 - #### Decision Tree Classifier Model
```py
ds=DecisionTreeClassifier(max_depth=3)
ds.fit(x_train,y_train)
train_pred=ds.predict(x_train)
test_pred=ds.predict(x_test)
print(accuracy_score(train_pred,y_train))
print(accuracy_score(test_pred,y_test))

import matplotlib.pyplot as plt
from  sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(ds,feature_names=x.columns.tolist(),class_names=["0","1"],filled=True)
plt.show()
```

###### Result 

![Img_15](https://github.com/Aathimuthu25/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/158067286/a2f200cc-6b57-46e9-9a92-25cd7a67f494)

- ### Conclusion

- We are having 9 variables in our dataset. We look for some null values and duplicate values but there is no null values and duplicates. After we detect the outliers using box and most of the column having huge number of outliers so we created 3 copies of our original data d1,d2,d3 and then we apply outlier treatment in all the all the valiables in d1. Then we have apply outlier trreatment in some variabls where evere having less ammount of variabels.Then we apply all teh Maching learning models.


- we have applied so many machine learning models to get the perfect result without overfitting, and finally we get the final model using Logistic regression model there we got 99% accuracy and there is no overfitting.
















