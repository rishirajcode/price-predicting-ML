#!/usr/bin/env python

#rishiraj

# In[31]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

boston = "load_boston"
import pandas as pd


# In[32]:



from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import HTML


# In[42]:


#load the data

boston=load_boston()

#description
print(boston.DESCR)

features=pd.DataFrame(boston.data,columns=boston.feature_names)
features


# In[46]:



features[ 'AGE']


target=pd.DataFrame(boston.target,columns=['target'])
target


# In[47]:


max(target['target'])


# In[49]:


min(target['target'])


# In[50]:

df=pd.concat([features,target],axis=1)
df
#given below is the concatinated dataframe


# In[53]:




df.describe().round(decimals=2)



# In66]:



corr=df.corr('pearson')

#absolute value of correlation

corrs=[abs(corr[attr]['target'])for attr in list(features)]

#list of pairs (corrs,features)
lt=list(zip(corrs, list(features)))


lt.sort(key=lambda x: x[0], reverse=True)


corrs,labels=list(zip((*lt)))


index= np.arange(len(labels))
plt.figure(figsize=(20,10))
plt.bar(index,corrs,width=0.5)
plt.xlabel("Attributes")
plt.ylabel('correlation')
plt.xticks(index,labels)
plt.show()


# In[67]:



X=df['LSTAT'].values
Y=df['target'].values


print(Y[:5])


# In[73]:




x_scaler=MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:,-1]

y_scaler=MinMaxScaler()
Y=y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:,-1]

print(Y[:5])


# In[75]:



def error (m,x,t,c):
    N=x.size
    e=sum(((m*x + c)-t)**2)
    return e*1/(2*N)


# In[76]:


#spliting the data
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)


# In[77]:


def error (m,x,t,c):
    N=x.size
    e=sum(((m*x + c)-t)**2)
    return e*1/(2*N)


# In[78]:


def update (m,c,x,t,learning_rate):
    grad_m=sum(2*((m*x +c)-t)*x)
    grad_c=sum(2*((m*x+c)-t))
    m=m-grad_m*learning_rate
    c=c-grad_c*learning_rate
    return m,c


# In[90]:


def gradient_descent(init_m,init_c,x,t,learning_rate,iterations,error_threshold):
    m=init_m
    c=init_c
    error_value=list()
    values_mc=list()
    for i in range(iterations):
        e=error(m,x,c,t)
        if e<error_threshold:
            print("Error less than the threshold. Stopping gradient descent")
            break
        error_value.append(e)
        m,c=update(m,x,c,t,learning_rate)
        values_mc.append(m)
    return m,c , error_value, values_mc
            


# In[91]:


#percentage timing

init_m=0.9
init_c=0
learning_rate=0.001
iterations=250
error_threshold=0.001

m,c,error_value,values_mc= gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)


# In[107]:



plt.scatter(xtrain,ytrain,color='y')

plt.plot(xtrain,(m*xtrain+c), color= 'b')


# In[98]:


#plotiing the error values
plt.plot(np.arange(len(error_value)),error_value)
plt.ylabel('Error')
plt.xlabel('iterations')


# In[147]:


#no of iterations is dirctly prop to change in lines

mc_values_anim= values_mc[0:200:5]


fig , ax=plt.subplots()
ln, =plt.plot([],[], 'ro-', animated=True)


def init():
    plt.scatter(xtest,ytest,color='r')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    return ln,

def update_frame(frame):
    m,c = mc_values_anim[frame]
    x1,y1=0.5,m *(-0.5) +c
    x2,y2=1.5, m*1.5+c
    ln.set_data([x1,x2],[y1,y2])
    return ln,

anim=FuncAnimation(fig,update_frame,frames= range(len(mc_values_anim)), init_func=init,blit=True)


# In[168]:


predicted = (m + xtest)


# In[170]:


mean_squared_error(ytest, predicted) 


# In[171]:


p=pd.DataFrame(list(zip(xtest,ytest,predicted)), columns=['x', 'target_y', 'predicted_y'])
p.head()


# In[173]:


plt.scatter(xtest,ytest,color='b')

plt.plot(xtest,predicted,color='r')


# In[176]:


p=pd.DataFrame(list(zip(xtest,ytest,predicted)),columns=['x','target_y','predicted_y'])
p=p.round(decimals=2)
p.head()






