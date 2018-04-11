cols = 81 #Number of columns(features) in train data file


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

#path for train data file
path_train = "/home/bafna/Desktop/ML/KAGGLE/house-prices-advanced-regression-techniques/my_train.csv";
path_test = "/home/bafna/Desktop/ML/KAGGLE/house-prices-advanced-regression-techniques/my_test.csv";
path_out = "/home/bafna/Desktop/ML/KAGGLE/house-prices-advanced-regression-techniques/output.csv";
#list of intergers(index) which we want to select or import from training file
l = []
for i in range(0,cols-1):
	l.append(i);





#print(l[:10])
'''
fetch the training data using genfromtxt mnethod of numpy
$$$$$ Attribute description $$$
path: defines the path of file which we want to read
delimiter: specify separator between two attribute(feature)
autostrip: remove front and back spaces from an feature value
missing_values : All the comma separated string consider as missing values
filling_values : All the missing values filled with 0 
usecols: tells whgich column to select from file(mentioned in pathy variable)
skip_header: tells that how many starting rows we have to skip
skip_footer: tells that how many last rows we have to skip
'''
'''
data = np.genfromtxt(path, delimiter=",", usecols=(l), autostrip=True, dtype=None,
					 missing_values=("N/A","None",""),
					 filling_values=0, 
					 skip_header=1, skip_footer=0)
'''
#data = np.genfromtxt(path, delimiter=",", dtype=None, usecols=(l), skip_header=1, encoding=None)

#data = np.recfromcsv(path, delimiter=",", dtype=None, usecols=(l), skip_header=1, encoding=None)

data_train = pd.read_csv(path_train, header=None,skiprows=1,usecols=l)
target = pd.read_csv(path_train, header=None,skiprows=1,usecols=[81])
data_test = pd.read_csv(path_test, header=None,skiprows=1,usecols=l)
target_test = pd.read_csv(path_test, header=None,skiprows=1,usecols=[81])

#data_train , data_test , target_train , target_test = train_test_split(data_train ,
				 #target , test_size=0.3, random_state=42)


print(target[:10])
print(target_test[:10])

print("Train file shape is : ",data_train.shape)
print("Test file shape is : ",data_test.shape)
#data.replace(np.nan,0,inplace=True)
'''
x = data.dtypes;
m,n = data.shape
cnt = 0;
for i in range(n):
	if x[i] == 'object':
		cnt = cnt+1
		#print("This is object....\n");
		data = pd.get_dummies(data, columns=[i]);

print(data.shape)
print("Number of categorical attribute is : ",cnt)
'''

m,n = data_train.shape
#print("M and N is : ",m,n)
cnt = 0;
x = data_train.dtypes
for i in range(n):
	if x[i] == 'object':
		cnt = cnt+1
		#print("for i is : ",i)
		data_train[i].replace(np.nan,"AAAA",inplace=True)
		data_test[i].replace(np.nan,"AAAA",inplace=True)
		lb_style = LabelBinarizer()
		lb_results_train = lb_style.fit_transform(data_train[i].astype(str))
		lb_results_test = lb_style.transform(data_test[i].astype(str))
		p,q = lb_results_train.shape
		if q != 1:
			data1_train = pd.DataFrame(lb_results_train, columns=lb_style.classes_)
			data1_test = pd.DataFrame(lb_results_test, columns=lb_style.classes_)

		else:
			data1_train = pd.DataFrame({"AAAA":lb_results_train[:,0]})
			data1_test = pd.DataFrame({"AAAA":lb_results_test[:,0]})


		data_train = pd.concat([data_train,data1_train],axis=1)
		data_test = pd.concat([data_test,data1_test],ignore_index=False,axis=1)
'''
print("Train file shape is : ",data_train.shape)
print("Test file shape is : ",data_test.shape)
print("Number of categorical attribute is : ",cnt)
print("Train file datatype is : ",data_train.dtypes)
print("Test file datatype is : ",data_test.dtypes)
'''
#print("Mid \n")
#print("Train file shape is : ",data_train.shape)
#print("Test file shape is : ",data_test.shape)

data_train.replace(np.nan,0,inplace=True)
data_test.replace(np.nan,0,inplace=True)
target.replace(np.nan,0,inplace=True)
del data_train[0];
for i in range(n):
	if x[i] == 'object':
		cnt = cnt+1
		del data_train[i];
		del data_test[i];
'''
print("Train file shape is : ",data_train.shape)
print("Test file shape is : ",data_test.shape)
#print("Number of categorical attribute is : ",cnt)
print("Train file datatype is : ",data_train.dtypes)
print("Test file datatype is : ",data_test.dtypes)
'''

data_train = data_train.as_matrix()
data_test = data_test.as_matrix()
target = target.as_matrix()
target_test = target_test.as_matrix()
#print(target)
#target = target.astype(int)

#print(target)

p,q = data_test.shape
ids = data_test[:,0:1]
data_test = data_test[:,1:q]

#print("Target data type is : ",target.dtype);
print("Train shape is : ",data_train.shape)
print("target shape is : ",target.shape)
print("test shape is: ",data_test.shape)


print("Number of missing values in train data is : ", np.count_nonzero(np.isnan(data_train)))

#data_train = (data_train - data_train.mean(axis=0))/data_train.std(axis=0)
#data_test = (data_test - data_test.mean(axis=0))/(data_test.std(axis=0)+1)
#target = (target - target.mean(axis=0))/target.std(axis=0)

regression = linear_model.LinearRegression()
regression.fit(data_train,target)
predicted = regression.predict(data_test)

err=0;
for i in range(438):
	err = err + (target_test[i,0]-predicted[i,0])**2;
print(err)

print("Error on my test is : ", sum((predicted[:,0:1] - target_test)**2))

#output = np.concatenate((ids,predicted),axis=1)
#np.savetxt(path_out,output,delimiter=",",header="Id,SalePrice", comments='', fmt='%d,%1.10e')