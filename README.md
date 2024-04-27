# <p align="center"> Bank-Loan-Defaulter-Prediction-Project-Using-Python </p>
 
# <p align="center">![image](https://github.com/AhamedSahil/Bank-Loan-Default-Prediction-Project-Using-Python/assets/164605797/5ebc19c1-d7fd-4287-aae0-267d1d10b400)

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

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/f67f1aba-b496-4191-a304-e7e15766a78a)

 - ##### We are not having any null values.

```py
#2.missing values treatment
data_ex.isnull().sum()
```

###### Result

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/b94bf840-eec4-49c8-aa27-7a1172899649)

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

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/367f6427-c3aa-467f-9b23-e9b45fcb2669)

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/adfe04cb-7e05-4fee-9ac4-40c5effc185e)

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/3aa4322b-a8e8-4a6b-b0a8-c4d27bb33784)

- #### checking the distribution of the data

```py
#4.checking the distribution of the data
#Using random distribution 
for col in d1.columns:
    sns.displot(d1[col],kde = True)
```

###### Result:

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/937d674f-8b27-43f3-9b99-c07c1ff06b9c) 
![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/c7ae5608-5dfd-435c-b8b1-910536cd5119)

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/c5a6dbc6-857c-4368-9be0-8678a3c552bc)
![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/86dae432-d6dc-4653-863d-d9d7d2991b84)

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/574280df-fa05-427b-8cb7-4931b76a52c3)
![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/12e87db6-c3ee-468b-9341-82eea802850a) 

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/345d1d2d-82b6-4303-8a52-44afbc6284cf) 
![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/b649749a-4c23-4945-84aa-5dccb9ea5fa8) 

- #### Correlation 
```py
#6.correlation analysis 
d1.corr()
sns.heatmap(d1.corr().abs(),annot=True)
plt.show()
```
![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/8684ed71-474b-4ba5-9098-438cc531c54d)

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

![image](https://github.com/AhamedSahil/Bank-Loan-Defaulter-Prediction-Project-Using-Python/assets/164605797/01759251-e2ce-49a0-9868-a2607cb1509b)

- ### Conclusion

- We are having 9 variables in our dataset. We look for some null values and duplicate values but there is no null values and duplicates. After we detect the outliers using box and most of the column having huge number of outliers so we created 3 copies of our original data d1,d2,d3 and then we apply outlier treatment in all the all the valiables in d1. Then we have apply outlier trreatment in some variabls where evere having less ammount of variabels.Then we apply all teh Maching learning models.


- we have applied so many machine learning models to get the perfect result without ofverfitting, and finally we get the final model using Logistic regression model there we got 99% accuracy and there is no overfitting.
















