#!/usr/bin/env python
# coding: utf-8

# # **加载模块，读取数据**
# 

# In[1]:


#所有需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch
from torch import nn
import torch.nn.functional as Fun

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']#Times New Roman
plt.rcParams['axes.unicode_minus'] = False
#解决负号显示问题
plt.rcParams['axes.unicode_minus']=False

#data:原始数据
#     '''
#     这里的dataname可以有八口井可选
#     PG101-3 102-2 201-1 201-2 301-3 302-3 2011-5 3011-5
#     '''

dataname='PG102-2.csv'
pdread=dataname#+'.csv'
data=pd.read_csv(pdread,encoding='gb2312')
#数据表格展示
data

import torch

DEVICE = torch.device( "cpu")#"cuda" if torch.cuda.is_available() else
print('You are using: ' + str(DEVICE))


# # 先行计算

# ## 计算日产气能、日产水能

# In[2]:


#g_ene日产气能

g_ene=data['日产混合气']/data['生产时间']
g_ene.isna().sum()


# In[3]:


#w_ene日产水能

w_ene=data['日产水']/data['生产时间']
w_ene.isna().sum()


# ## 填充缺失值，组成数据集

# In[4]:


#查看有默认标记的异常值
data.iloc[data.values == '?'].count()


# In[5]:


#缺失值填充
def fill_missing(values):
    '''
    该函数实现缺失值填充
    思路：将前一天同一时刻的采样值用来填充缺失值
    '''
    gap = 7
    for row in range(values.shape[0]):# 行循环
        if np.isnan(values[row]):
            values[row] = np.mean(values[row - gap-1:row])

# def outlier_replace(values):
#     gap = 7
#     for row in range(values.shape[0]):# 行循环
#         mu=np.mean(values[row - gap-1:row])
#         sigma=
#         if np.isnan(values[row]):
#             values[row] = np.mean(values[row - gap-1:row])
        


# In[6]:


fill_missing(g_ene.values) # 填充缺失值
fill_missing(w_ene.values)


# In[7]:


data['日产气能']=g_ene
data['日产水能']=w_ene


# In[8]:


data


# In[9]:


#再次检测异常值
data.isna().sum()


# # 生成训练数据集

# In[10]:


#取前三列数据

#data_nh：只保留数值数据
#raw_data：numpy数值数据

data_nh=data.iloc[:,3:17]
raw_data=data_nh.to_numpy()
data_nh


# In[11]:


#绘制图像

#线图

plt.figure(figsize=(20,5))
plt.ylabel('生产时间',fontsize=20)
data[ '生产时间' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('油温',fontsize=20)
data[ '油温' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('套温',fontsize=20)
data[ '套温' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('油压',fontsize=20)
data[ '油压' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('套压',fontsize=20)
data[ '套压' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('日产混合气',fontsize=20)
data[ '日产混合气' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('日产水',fontsize=20)
data[ '日产水' ].plot(color='deepskyblue')



plt.figure(figsize=(20,5))
plt.ylabel('日产气能',fontsize=20)
data[ '日产气能' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('日产水能',fontsize=20)
data[ '日产水能' ].plot(color='deepskyblue')


# In[12]:


#散点图

plt.figure(figsize=(20,5))
plt.ylabel('生产时间',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,0])),data_nh.iloc[:,0],c='deepskyblue',marker='.')

plt.figure(figsize=(20,5))
plt.ylabel('油温',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,1])),data_nh.iloc[:,1],c='deepskyblue',marker='.')

plt.figure(figsize=(20,5))
plt.ylabel('套温',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,2])),data_nh.iloc[:,2],c='deepskyblue',marker='.')

plt.figure(figsize=(20,5))
plt.ylabel('油压',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,3])),data_nh.iloc[:,3],c='deepskyblue',marker='.')

plt.figure(figsize=(20,5))
plt.ylabel('套压',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,4])),data_nh.iloc[:,4],c='deepskyblue',marker='.')

plt.figure(figsize=(20,5))
plt.ylabel('日产混合气',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,5])),data_nh.iloc[:,5],c='deepskyblue',marker='.')

plt.figure(figsize=(20,5))
plt.ylabel('日产水',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,6])),data_nh.iloc[:,6],c='deepskyblue',marker='.')



plt.figure(figsize=(20,5))
plt.ylabel('日产气能',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,12])),data_nh.iloc[:,12],c='deepskyblue',marker='.')

plt.figure(figsize=(20,5))
plt.ylabel('日产水能',fontsize=20)
plt.scatter(range(0,len(data_nh.iloc[:,13])),data_nh.iloc[:,13],c='deepskyblue',marker='.')


# In[13]:


#绘制分布图——日产气能

plt.figure()
plt.figure(figsize=(20,5))
plt.hist(data_nh.iloc[:,12], bins=80, histtype="stepfilled", alpha=.8,color='deepskyblue')


# # 数据清洗

# In[14]:


#异常值检测

data_prow=7

import pandas as pd
from matplotlib import pyplot

# Taking moving average of last 6 obs
rolling = data_nh.iloc[:,12].rolling(window=data_prow)
rolling_mean = rolling.mean()

rolling_mean[:data_prow]=data_nh.iloc[:,12][:data_prow]

rolling_std=np.nan_to_num(rolling_mean.rolling(window=data_prow).var()**0.5)
# plot the two series


plt.figure(figsize=(30,10))
plt.ylabel('日产气能',fontsize=20)
pyplot.plot(data_nh.iloc[:,12],color='deepskyblue')
pyplot.plot(rolling_mean, color='red')
rup=rolling_mean+3*rolling_std
rdown=rolling_mean-3*rolling_std
pyplot.plot(rup, color='coral')
pyplot.plot(rdown, color='coral')
#图例
plt.legend(['True','Predict','Range'])
plt.tick_params(labelsize=20)
pyplot.show()



rolling_mean,rolling_std


# In[15]:


EN=data_nh.iloc[:,12]
A=EN>rup
B=EN<rdown
C=A^B
A.astype(int)
B.astype(int)
C.astype(int)
A.sum(),B.sum(),C.sum(),C


# In[16]:


EN[1]


# In[17]:


ENN=[]
t=0
for i in C:
    if i==True:
        ENN.append(rolling_mean[t])
    else:
        ENN.append(EN[t])
    t+=1


# In[18]:


plt.figure(figsize=(30,10))

pyplot.plot(rolling_mean, color='red')
rup=rolling_mean+3*rolling_std
rdown=rolling_mean-3*rolling_std
pyplot.plot(rup, color='coral')
pyplot.plot(rdown, color='coral')
pyplot.plot(ENN,color='deepskyblue')
#图例
plt.legend(['True','Predict','Range'])
pyplot.show()


# In[19]:


plt.figure(figsize=(30,10))
pyplot.plot(data_nh.iloc[:,12],color='red')
pyplot.plot(ENN,color='deepskyblue')
plt.ylabel('日产气能',fontsize=20)
plt.tick_params(labelsize=20)
plt.legend(['Raw','Processed'])


# In[20]:


D_P_ERROR=data_nh.iloc[:,12]-ENN
D_P_ERROR,D_P_ERROR.sum()/len(D_P_ERROR)/np.mean(ENN),'可接受'


# In[21]:


data_nh.iloc[:,12]=ENN
data_nh.iloc[:,12]==ENN


# In[22]:


data_nh


# # 数据归一化和相关性分析

# In[23]:


#标准化处理
data_nh_std=data_nh.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))

#data_nh_std:标准化数据

data_nh_std


# In[24]:


#存储标准化数据
data_nh_std.to_csv(dataname+'_a'+'.csv')


# In[25]:


#相关性分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 探索性数据分析 EDA
# 最简单粗暴的方式就是根据 HeatMap 热力图分析各个指标之间的关联性
f, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(data_nh_std.corr(), fmt="d", linewidths=0.5, ax=ax,cmap='RdBu')#bwrcool\winter
plt.show()

# corr = data_nh.corr()
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# with sns.axes_style("white"):
#     f, ax = plt.subplots(figsize=(9, 8))
#     sns.heatmap(corr, mask=mask, vmax=.3, fmt="d", linewidths=0.5, ax=ax,cmap='cool')


# In[26]:


#划分数据集

def split_dataset(data):
    '''
    该函数实现以周为单位切分训练数据和测试数据
    '''
    # data为按天的耗电量统计数据，shape为(1442, 8)
    # 测试集取最后一年的46周（322天）数据，剩下的159周（1113天）数据为训练集，以下的切片实现此功能。
    train, test = data[1:-323], data[-323:-1]
#     train = np.array(np.split(train, len(train)/7)) # 将数据划分为按周为单位的数据
#     test = np.array(np.split(test, len(test)/7))
    return train, test


def num(data):
    a=list(data).index('日产气能')
    return a

def sliding_window(train, sw_width=7, n_out=7, in_start=0):
    '''
    该函数实现窗口宽度为7、滑动步长为7的滑动窗口截取序列数据
    '''
    a=num(train)
    train=train.values
    data=train
    X, y = [], []

    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            X.append(data[in_start:in_end, 0:a+1])
            y.append(data[in_end:out_end,a])

        in_start += sw_width

    return np.array(X), np.array(y)

def data_processing(sw_width=7,n_out=7,data=data_nh_std):
    A,test=split_dataset(data)
    train,valid=split_dataset(A)
    train=train.iloc[7:,:]
    train_X,train_y=sliding_window(train,sw_width,n_out)
    test_X,test_y=sliding_window(test,sw_width,n_out)
    valid_X,valid_y=sliding_window(valid,sw_width,n_out)


    print('\n\ntrain X size =')
    print(train_X.shape)

    print('\n\ntrain y size =\n')
    print(train_y.shape)

    print('\n\ntest X =\n')
    print( test_X)
    print('\n\ntest X size =\n')
    print(test_X.shape)

    print('\n\ntest y =\n')
    print(test_y)
    print('\n\ntest y size =\n')
    print(test_y.shape)
    return train,test,valid,train_X,train_y,test_X,test_y,valid_X,valid_y
    

sw_width=7    
n_out=7    
train,test,valid,train_X,train_y,test_X,test_y,valid_X,valid_y=data_processing(sw_width,n_out)


# In[27]:


#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#数据集划分方式展示

plt.figure(figsize=(20,5))
plt.ylabel('生产时间',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '生产时间' ].plot(color='deepskyblue')


plt.figure(figsize=(20,5))
plt.ylabel('油温',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '油温' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('套温',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '套温' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('油压',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '油压' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('套压',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '套压' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('日产混合气',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '日产混合气' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('日产水',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '日产水' ].plot(color='deepskyblue')



plt.figure(figsize=(20,5))
plt.ylabel('日产气能',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '日产气能' ].plot(color='deepskyblue')

plt.figure(figsize=(20,5))
plt.ylabel('日产水能',fontsize=20)
plt.xticks(range(0,13*7,7))
plt.xlim(-7,59)
data_nh_std[ '日产水能' ].plot(color='deepskyblue')


# # 时间序列方法 

# ## ARMA

# In[28]:


# data=pd.read_csv(pdread,encoding='gb2312',index_col=u'日期')
data_ar=data_nh_std.iloc[:,12:13]

data_ar


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data_ar.plot()
plt.show()


# In[30]:


data_ar.isna().sum()


# In[31]:


# 平稳性检测
# 自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data_ar).show()

# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print(u'原始序列的ADF检验结果为：', ADF(data_ar[u'日产气能']))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore


# In[32]:


# 差分后的结果
D_data = data_ar.diff().dropna()
D_data.columns = [u'日产气能差分']
D_data.plot()  # 时序图
plt.show()
plot_acf(D_data).show()  # 自相关图
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data).show()  # 偏自相关图
print(u'差分序列的ADF检验结果为：', ADF(D_data[u'日产气能差分']))  # 平稳性检测

# 白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值


# In[33]:


from statsmodels.tsa.arima_model import ARMA
AR_result_origin=[]
a=pd.Series(data_ar['日产气能'].values[:-322*2+0*7])
my_arma = ARMA(a, (1, 0))  # 这里的(1, 0)从arma_order_select_ic函数返回,但是这里返回6,7运行失败
model = my_arma.fit()
ARR = model.forecast(322*2)[0]
for j in ARR:
    AR_result_origin.append(j)

AR_result_origin = np.asarray(AR_result_origin)
print('result:', AR_result_origin)

from sklearn.metrics import mean_squared_error 
ARMSE=mean_squared_error(AR_result_origin,data_ar['日产气能'].values[-322-322:])
print('MSE:',ARMSE)


# In[34]:


plt.figure(figsize=(30,10))

plt.plot(data_ar['日产气能'].values,color='deepskyblue')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], AR_result_origin)),color='coral')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], AR_result_origin))-data_ar['日产气能'].values,color='aquamarine')

#分割线
plt.axvline(len(data_ar['日产气能'].values)-322,c='violet')
plt.axvline(len(data_ar['日产气能'].values)-322-322,c='violet')
plt.tick_params(labelsize=20)
#图例
plt.legend(['True','Predict','Error'])


# In[35]:


from statsmodels.tsa.arima_model import ARMA
AR_result=[]
for i in range(46*2):
    a=pd.Series(data_ar['日产气能'].values[:-322*2+i*7])
    my_arma = ARMA(a, (1, 0))  # 这里的(1, 0)从arma_order_select_ic函数返回,但是这里返回6,7运行失败
    model = my_arma.fit()
    ARR = model.forecast(7)[0]
    for j in ARR:
        AR_result.append(j)

AR_result = np.asarray(AR_result)
print('result:', AR_result)

from sklearn.metrics import mean_squared_error 
ARMSE=mean_squared_error(AR_result,data_ar['日产气能'].values[-322-322:])
print('MSE:',ARMSE)


# In[36]:


plt.figure(figsize=(30,10))

plt.plot(data_ar['日产气能'].values,color='deepskyblue')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], AR_result)),color='coral')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], AR_result))-data_ar['日产气能'].values,color='aquamarine')

#分割线
plt.axvline(len(data_ar['日产气能'].values)-322,c='violet')
plt.axvline(len(data_ar['日产气能'].values)-322-322,c='violet')
plt.tick_params(labelsize=20)
#图例
plt.legend(['True','Predict','Error'])


# ## ARIMA

# In[37]:


from statsmodels.tsa.arima_model import ARIMA
ARI_result_origin=[]

a=pd.Series(data_ar['日产气能'].values[:-322*2+0*7])
my_arima = ARIMA(a, (7,1,0))  # 这里的(1, 0)从arma_order_select_ic函数返回,但是这里返回6,7运行失败
model = my_arima.fit()
ARIR = model.forecast(322*2)[0]
for j in ARIR:
    ARI_result_origin.append(j)

ARI_result_origin = np.asarray(ARI_result_origin)
print('result:', ARI_result_origin)

from sklearn.metrics import mean_squared_error 
ARIMSE=mean_squared_error(ARI_result_origin,data_ar['日产气能'].values[-322-322:])
print('MSE:',ARIMSE)


# In[38]:


plt.figure(figsize=(30,10))

plt.plot(data_ar['日产气能'].values,color='deepskyblue')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], ARI_result_origin)),color='coral')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], ARI_result_origin))-data_ar['日产气能'].values,color='aquamarine')

#分割线
plt.axvline(len(data_ar['日产气能'].values)-322,c='violet')
plt.axvline(len(data_ar['日产气能'].values)-322-322,c='violet')
plt.tick_params(labelsize=20)
#图例
plt.legend(['True','Predict','Error'])


# In[39]:


from statsmodels.tsa.arima_model import ARIMA
ARI_result=[]
for i in range(46*2):
    a=pd.Series(data_ar['日产气能'].values[:-322*2+i*7])
    my_arima = ARIMA(a, (7,1,0))  # 这里的(1, 0)从arma_order_select_ic函数返回,但是这里返回6,7运行失败
    model = my_arima.fit()
    ARIR = model.forecast(7)[0]
    for j in ARIR:
        ARI_result.append(j)

ARI_result = np.asarray(ARI_result)
print('result:', ARI_result)

from sklearn.metrics import mean_squared_error 
ARIMSE=mean_squared_error(ARI_result,data_ar['日产气能'].values[-322-322:])
print('MSE:',ARIMSE)


# In[40]:


plt.figure(figsize=(30,10))

plt.plot(data_ar['日产气能'].values,color='deepskyblue')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], ARI_result)),color='coral')

plt.plot(np.hstack((data_ar['日产气能'].values[:-322-322], ARI_result))-data_ar['日产气能'].values,color='aquamarine')

#分割线
plt.axvline(len(data_ar['日产气能'].values)-322,c='violet')
plt.axvline(len(data_ar['日产气能'].values)-322-322,c='violet')
plt.tick_params(labelsize=20)
#图例
plt.legend(['True','Predict','Error'])


# # 神经网络方法
# 

# ## 数据处理\绘图

# In[41]:


#数据扁平化
n_input = train_X.shape[1] * train_X.shape[2]
def datalining(train_X=train_X,valid_X=valid_X,test_X=test_X):
    MLP_trainX = train_X.reshape((train_X.shape[0], n_input))
    MLP_testX = test_X.reshape((test_X.shape[0], n_input))
    MLP_validX=valid_X.reshape((valid_X.shape[0],n_input))
    return MLP_trainX,MLP_validX,MLP_testX
MLP_trainX,MLP_validX,MLP_testX=datalining()    
train_size=train_y.shape[0]*train_y.shape[1]
valid_size=valid_y.shape[0]*valid_y.shape[1]
test_size=test_y.shape[0]*test_y.shape[1]

#对train、valid、test计算长度
trainstep=np.linspace(0, train_size, train_size, dtype=np.float32)
validstep=np.linspace(train_size,train_size+len(valid),len(valid)-sw_width-n_out,dtype=np.float32)
teststep=np.linspace(train_size+valid_size,train_size+valid_size+len(test),len(test)-sw_width-n_out,dtype=np.float32)


# In[42]:


#数据的展示

plt.figure(figsize=(20,5))
plt.ylabel('日产气能',fontsize=20)
data_nh[ '日产气能' ].plot(color='deepskyblue')
#分割线
plt.axvline(train_size,c='violet')
plt.axvline(train_size+valid_size,c='violet')


# In[43]:


#数据tensor化
def data2tensor(MLP_trainX=MLP_trainX,
                train_y=train_y,
                MLP_validX=MLP_validX,
                valid_y=valid_y,
                MLP_testX=MLP_testX,
                test_y=test_y
                ):
            
    train_x_tensor = torch.tensor(MLP_trainX).reshape(1,np.size(MLP_trainX,0),np.size(MLP_trainX,1))
    train_y_tensor = torch.tensor(train_y).reshape(1,np.size(train_y,0),np.size(train_y,1))

    # transfer data to pytorch tensor
    train_x_tensor = train_x_tensor.to(torch.float32)
    train_y_tensor = train_y_tensor.to(torch.float32)
    # valid_x_tensor = torch.from_numpy(valid_x)
    
    # prediction on valid dataset
    valid_x_tensor = torch.tensor(MLP_validX).reshape(1,np.size(MLP_validX,0),np.size(MLP_validX,1))
    valid_y_tensor = torch.tensor(valid_y).reshape(1,np.size(valid_y,0),np.size(valid_y,1))
    valid_x_tensor = valid_x_tensor.to(torch.float32)
    valid_y_tensor = valid_y_tensor.to(torch.float32)  
    
    # prediction on test dataset
    test_x_tensor = torch.tensor(MLP_testX).reshape(1,np.size(MLP_testX,0),np.size(MLP_testX,1))
    test_y_tensor = torch.tensor(test_y).reshape(1,np.size(test_y,0),np.size(test_y,1))
    test_x_tensor = test_x_tensor.to(torch.float32)
    test_y_tensor = test_y_tensor.to(torch.float32)
    
    return train_x_tensor,train_y_tensor,valid_x_tensor,valid_y_tensor,test_x_tensor,test_y_tensor


# In[44]:


#data变tensor
train_x_tensor,train_y_tensor,valid_x_tensor,valid_y_tensor,test_x_tensor,test_y_tensor=data2tensor()


# In[45]:


#计算残差
def residcount(predictive_y_for_training,
               predictive_y_for_validing,
               predictive_y_for_testing,
               train_y_tensor,
               valid_y_tensor,
               test_y_tensor
               ):
    #train段
    deltaytrain=predictive_y_for_training-train_y_tensor.view(-1, 1).data.numpy()
    
    #vaild段
    deltayvalid=predictive_y_for_validing-valid_y_tensor.view(-1, 1).data.numpy()
    
    #test段
    deltaytest=predictive_y_for_testing-test_y_tensor.view(-1, 1).data.numpy()
    
    return deltaytrain,deltayvalid,deltaytest


# In[46]:


#损失下降曲线图
def lossdecrease(trainl,
                 validl,
                 totall
                 ):
    
    #自然刻度
    plt.figure(figsize=(10,10))
    # plt.axes(yscale = "log")
    plt.plot(trainl,color='deepskyblue')
    plt.plot(validl,color='coral')
    plt.plot(totall,color='lime')
    plt.legend(['Train Loss','valid Loss','Total Loss'])
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.xlim(-20, 320)
    plt.ylim(0.001, 1.20)
    
    #半对数
    plt.figure(figsize=(10,10))
    plt.axes(yscale = "log")
    plt.plot(trainl,color='deepskyblue')
    plt.plot(validl,color='coral')
    plt.plot(totall,color='lime')
    plt.legend(['Train Loss','valid Loss','Total Loss'])
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.xlim(-20, 320)
    plt.ylim(0.001, 1.20)


# In[47]:


#预测结果图
def allplots(trainstep,
             validstep,
             teststep,
             train_y_tensor,
             valid_y_tensor,
             test_y_tensor,
             predictive_y_for_training,
             predictive_y_for_validing,
             predictive_y_for_testing,
             deltaytrain,
             deltayvalid,
             deltaytest,
             train_size,
             valid_size,
             ):
    #总体图
    plt.figure(figsize=(30,10))    

    #train段
    plt.plot(trainstep,train_y_tensor.flatten(),color='deepskyblue')
    plt.plot(trainstep,predictive_y_for_training,color='coral')#output.flatten().detach().numpy()
    plt.plot(trainstep,deltaytrain,color='aquamarine')

    #valid段
    plt.plot(validstep,valid_y_tensor.flatten(),color='deepskyblue')
    plt.plot(validstep,predictive_y_for_validing,color='coral')#output.flatten().detach().numpy()
    plt.plot(validstep,deltayvalid,color='aquamarine')
    
    #test段
    plt.plot(teststep,test_y_tensor.flatten(),color='deepskyblue')
    plt.plot(teststep,predictive_y_for_testing,color='coral')#output.flatten().detach().numpy()
    plt.plot(teststep,deltaytest,color='aquamarine')
    
    #图例
    plt.legend(['True','Predict','Error'])

    #分割线
    plt.axvline(train_size,c='violet')
    plt.axvline(train_size+valid_size,c='violet')

    


# In[48]:


#残差分布图
def residualsplots(data1,
                   data2
                   ):
    plt.figure(figsize=(20,10))
    markerline, stemlines, baseline = plt.stem(data1, data2,basefmt='aquamarine')
    plt.setp(stemlines, linewidth=2, color='deepskyblue')     # set stems to random colors
    plt.setp(markerline, 'markerfacecolor', 'cornflowerblue')    # make points blue


# In[49]:


#箱线图
def boxplot(data1,
            title
            ):
    df = pd.DataFrame(abs(data1))
    df.plot.box(title=title)
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()


# In[50]:


#交叉分布图
def jointplot(data1,
              data2
              ):

    from pandas import DataFrame
    validpredict=data1.reshape(-1,)
    validreal=data2.view(-1, 1).data.numpy().reshape(-1,)
    data_array=np.hstack([validpredict,validreal]).reshape(-1,2)
    data_frame = DataFrame(data_array,index=None,columns = ['TestPredict','TestReal'])

    import seaborn as sns
    sns.set(style="white",font_scale=1.5)
    g = sns.jointplot(x='TestPredict',
                      y='TestReal', 
                      data=data_frame,
                      color='deepskyblue',#修改颜色
                      #kind='reg'
                     )
    


# In[51]:


def allfigues(predictive_y_for_training,
             predictive_y_for_validing,
             predictive_y_for_testing,
             deltaytrain,
             deltayvalid,
             deltaytest,
             name
             ):
    #绘制总的预测图
    allplots(trainstep,
             validstep,
             teststep,
             train_y_tensor,
             valid_y_tensor,
             test_y_tensor,
             predictive_y_for_training,
             predictive_y_for_validing,
             predictive_y_for_testing,
             deltaytrain,
             deltayvalid,
             deltaytest,
             train_size,
             valid_size,
             )
   
    #残差分布图
    residualsplots(data1=validstep,
                   data2=deltayvalid
                   )
    
    
    #箱线图
    boxplot(deltayvalid,
            name,
            )
    
    #交叉分布图
    jointplot(data1=predictive_y_for_testing,
              data2=valid_y_tensor
              )


# ## 模型

# In[52]:


# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        #定义层
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.relu1 = nn.ReLU() #nn.Linear为线性关系，加上激活函数转为非线性
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# In[53]:


# 定义RNN神经网络
class RNNnet(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,n_output):
        super(RNNnet, self).__init__()
        self.rnn = nn.RNN(
        input_size,
        hidden_size, 
        num_layers, 
        batch_first=True,
        )
        self.out = nn.Linear(hidden_size, n_output)
    def forward(self, x, h_state):
         # x (batch, time_step, input_size)
         # h_state (n_layers, batch, hidden_size)
         # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [] # 保存所有的预测值
        for time_step in range(r_out.size(1)): # 计算每一步长的预测值
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


# In[54]:


# 定义RNN神经网络
class GRUnet(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,n_output):
        super(GRUnet, self).__init__()
        self.rnn = nn.GRU(
        input_size,
        hidden_size, 
        num_layers, 
        batch_first=True,
        )
        self.out = nn.Linear(hidden_size, n_output)
    def forward(self, x, h_state):
         # x (batch, time_step, input_size)
         # h_state (n_layers, batch, hidden_size)
         # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [] # 保存所有的预测值
        for time_step in range(r_out.size(1)): # 计算每一步长的预测值
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


# In[55]:


#解决中文显示问题
plt.rcParams['font.sans-serif']=['Times New Roman']#Times New Roman
plt.rcParams['axes.unicode_minus'] = False
#解决负号显示问题
plt.rcParams['axes.unicode_minus']=False


# In[56]:


def ANN(epoches=100,
        LR=0.003,
        train_x_tensor=train_x_tensor,
        train_y_tensor=train_y_tensor,
        valid_x_tensor=valid_x_tensor,
        valid_y_tensor=valid_y_tensor,
        test_x_tensor=test_x_tensor,
        test_y_tensor=test_y_tensor,
       ):
    
    #model建立
    NN_Model = Net(n_feature=n_input, n_hidden=200, n_output=train_y.shape[1])
    optimizer = torch.optim.Adam(NN_Model.parameters(), lr=LR)
    print('ANN model:', NN_Model)
    print('model.parameters:', NN_Model.parameters)  
    
    #model训练
    loss_function = nn.MSELoss()
    max_epochs = epoches
    trainl=[]
    validl=[]
    totall=[]

    for epoch in range(max_epochs):

        output = NN_Model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1)%10==0: #每训练10个批次可视化一下效果，并打印一下loss
            print("Train:")
            plt.plot(trainstep, train_y.flatten(),color='deepskyblue')
            plt.plot(trainstep[sw_width:], output.data.numpy().flatten()[sw_width:],color='aquamarine')
            plt.draw()
            plt.pause(0.01)
            print("valid:")
            plt.plot(validstep,torch.tensor(valid_y).to(torch.float32).flatten(),color='deepskyblue')
            plt.plot(validstep,predictive_y_for_validing,color='aquamarine')
            plt.pause(0.01)

        #train loss
        predictive_y_for_trainingo=NN_Model(train_x_tensor)
        trainloss=loss_function(predictive_y_for_trainingo,train_y_tensor)
        
        #valid loss
        ANN_valid_model = NN_Model.eval() # switch to validing model
        predictive_y_for_validingo = ANN_valid_model(valid_x_tensor)
        predictive_y_for_validing = predictive_y_for_validingo.view(-1, 1).data.numpy()
        
        validloss=loss_function(predictive_y_for_validingo, valid_y_tensor)
        
        totalloss=trainloss+validloss
        print('Epoch [{}/{}],trainloss: {:.4f},validloss: {:.4f},totalloss: {:.4f}'.format(epoch+1,max_epochs,trainloss,validloss,totalloss))
        trainl.append(trainloss.item())
        validl.append(validloss.item())
        totall.append(totalloss.item())

        
    #损失下降图
    lossdecrease(trainl,
                 validl,
                 totall
                 )
    
    #model测试    
    
    #train段
    predictive_y_for_trainingo = NN_Model(train_x_tensor)
    
    predictive_y_for_training = predictive_y_for_trainingo.view(-1, 1).data.numpy()
    
    #vaild段
    ANN_valid_model = NN_Model.eval() # switch to validing model
    predictive_y_for_validingo = ANN_valid_model(valid_x_tensor)
    
    predictive_y_for_validing = predictive_y_for_validingo.view(-1, 1).data.numpy()  
    
    #test段
    ANN_test_model = NN_Model.eval() # switch to testing model
    predictive_y_for_testingo = ANN_test_model(test_x_tensor)
    
    predictive_y_for_testing = predictive_y_for_testingo.view(-1, 1).data.numpy()
    
    #残差计算
    deltaytrain,deltayvalid,deltaytest=residcount(predictive_y_for_training,
               predictive_y_for_validing,
               predictive_y_for_testing,
               train_y_tensor,
               valid_y_tensor,
               test_y_tensor
               )
    
    #计算Loss并打印
    trainloss=loss_function(predictive_y_for_trainingo, train_y_tensor)    
    validloss=loss_function(predictive_y_for_validingo, valid_y_tensor)    
    testloss=loss_function(predictive_y_for_testingo, test_y_tensor)
    
    print("TrainLoss:{:4f}".format(trainloss))    
    print("ValidLoss:{:4f}".format(validloss))    
    print("TestLoss:{:4f}".format(testloss)) 

    allfigues(predictive_y_for_training,
             predictive_y_for_validing,
             predictive_y_for_testing,
             deltaytrain,
             deltayvalid,
             deltaytest,
             name='ANN')


# In[57]:


def RNN_fit(name,
        epoches=200,
        LR=0.0005,
        train_x_tensor=train_x_tensor,
        train_y_tensor=train_y_tensor,
        valid_x_tensor=valid_x_tensor,
        valid_y_tensor=valid_y_tensor,
        test_x_tensor=test_x_tensor,
        test_y_tensor=test_y_tensor,
        ):

    h_state=None

    #model建立    

    if name == 'simple RNN':
        NN_Model = RNNnet(input_size=n_input, hidden_size=200, num_layers=3,n_output=train_y.shape[1])
    elif name == 'GRU':
        NN_Model = GRUnet(input_size=n_input, hidden_size=200, num_layers=3,n_output=train_y.shape[1])
        
    optimizer = torch.optim.Adam(NN_Model.parameters(),lr=LR)   # Adam优化，几乎不用调参
    print('RNN model:', NN_Model)
    print('model.parameters:', NN_Model.parameters) 
    
    #model训练
    loss_function = nn.MSELoss()
    max_epochs = epoches
    trainl=[]
    validl=[]
    totall=[]
    NN_Model.train

    for epoch in range(max_epochs):

        output ,h_state = NN_Model(train_x_tensor,h_state)
        loss = loss_function(output, train_y_tensor)
        h_state = h_state.data
        h_state_valid=h_state
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1)%10==0: #每训练10个批次可视化一下效果，并打印一下loss
            print("Train:")
            plt.plot(trainstep, train_y.flatten(),color='deepskyblue')
            plt.plot(trainstep[sw_width:], output.data.numpy().flatten()[sw_width:],color='aquamarine')
            plt.draw()
            plt.pause(0.01)
            print("valid:")
            plt.plot(validstep,torch.tensor(valid_y).to(torch.float32).flatten(),color='deepskyblue')
            plt.plot(validstep,predictive_y_for_validing,color='aquamarine')
            plt.pause(0.01)

        #train loss
        predictive_y_for_trainingo,h_state_train=NN_Model(train_x_tensor,h_state)
        
        trainloss=loss_function(predictive_y_for_trainingo,train_y_tensor)
        
        #valid loss
        NN_valid_model = NN_Model.eval() # switch to validing model
        predictive_y_for_validingo ,h_state_valid= NN_valid_model(valid_x_tensor,h_state_valid)
        predictive_y_for_validing = predictive_y_for_validingo.view(-1, 1).data.numpy()
        
        validloss=loss_function(predictive_y_for_validingo, valid_y_tensor)
        
        totalloss=trainloss+validloss
        print('Epoch [{}/{}],trainloss: {:.4f},validloss: {:.4f},totalloss: {:.4f}'.format(epoch+1,max_epochs,trainloss,validloss,totalloss))
        trainl.append(trainloss.item())
        validl.append(validloss.item())
        totall.append(totalloss.item())

        
    #损失下降图
    lossdecrease(trainl,
                 validl,
                 totall
                 )
    
    #model测试    
    #train段
    predictive_y_for_trainingo,h_state_train = NN_Model(train_x_tensor,h_state)
    
    predictive_y_for_training = predictive_y_for_trainingo.view(-1, 1).data.numpy()
    #vaild段
    NN_valid_model = NN_Model.eval() # switch to validing model
    predictive_y_for_validingo ,h_state_valid= NN_valid_model(valid_x_tensor,h_state)
    
    predictive_y_for_validing = predictive_y_for_validingo.view(-1, 1).data.numpy()  
    #test段
    NN_test_model = NN_Model.eval() # switch to testing model
    predictive_y_for_testingo ,h_state_test= NN_test_model(test_x_tensor,h_state)
    
    predictive_y_for_testing = predictive_y_for_testingo.view(-1, 1).data.numpy()
    
    #残差计算
    deltaytrain,deltayvalid,deltaytest=residcount(predictive_y_for_training,
               predictive_y_for_validing,
               predictive_y_for_testing,
               train_y_tensor,
               valid_y_tensor,
               test_y_tensor
               )
    
    #计算Loss并打印
    trainloss=loss_function(predictive_y_for_trainingo, train_y_tensor)    
    validloss=loss_function(predictive_y_for_validingo, valid_y_tensor)    
    testloss=loss_function(predictive_y_for_testingo, test_y_tensor)
    
    print("TrainLoss:{:4f}".format(trainloss))    
    print("ValidLoss:{:4f}".format(validloss))    
    print("TestLoss:{:4f}".format(testloss)) 

    #绘制总的预测图
    allfigues(predictive_y_for_training,
             predictive_y_for_validing,
             predictive_y_for_testing,
             deltaytrain,
             deltayvalid,
             deltaytest,
             name)


# In[58]:


ANN(epoches=100,LR=0.004)


# In[59]:


RNN_fit('simple RNN',
        epoches=130,
        LR=0.004)


# In[60]:


RNN_fit('GRU',
        epoches=130,
        LR=0.004)


# # Attention

# In[107]:


class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention


q = train_y_tensor.reshape(-1,1,7)
k = train_x_tensor.reshape(-1,13,7)
v = k
attention = dot_attention()
context, attention = attention(q, k, v)
a=np.mean(attention.numpy(),axis=0)

MLP_trainX_att=train_x_tensor.reshape(-1,7,13)*torch.tensor(a)
MLP_trainX_att=MLP_trainX_att.reshape(-1,91).numpy()
MLP_testX_att=test_x_tensor.reshape(-1,7,13)*torch.tensor(a)
MLP_testX_att=MLP_testX_att.reshape(-1,91).numpy()
MLP_validX_att=valid_x_tensor.reshape(-1,7,13)*torch.tensor(a)
MLP_validX_att=MLP_validX_att.reshape(-1,91).numpy()
MLP_testX_att


# In[105]:


train_x_tensor_att,train_y_tensor,valid_x_tensor_att,valid_y_tensor,test_x_tensor_att,test_y_tensor=data2tensor(MLP_trainX=MLP_trainX_att,
                                                                                                                train_y=train_y,
                                                                                                                MLP_validX=MLP_validX_att,
                                                                                                                valid_y=valid_y,
                                                                                                                MLP_testX=MLP_testX_att,
                                                                                                                test_y=test_y
                                                                                                                )


# In[108]:


ANN(epoches=100,
    LR=0.01,
    train_x_tensor=train_x_tensor_att,
    train_y_tensor=train_y_tensor,
    valid_x_tensor=valid_x_tensor_att,
    valid_y_tensor=valid_y_tensor,
    test_x_tensor=test_x_tensor_att,
    test_y_tensor=test_y_tensor,
   )


# In[65]:


RNN_fit('GRU',
        epoches=100,
        LR=0.01,
        train_x_tensor=train_x_tensor,#attention
        train_y_tensor=train_y_tensor,
        valid_x_tensor=valid_x_tensor,#attention
        valid_y_tensor=valid_y_tensor,
        test_x_tensor=test_x_tensor,#attention
        test_y_tensor=test_y_tensor,
        )


# # 引入噪声

# ## 引入合成噪声

# In[66]:


import numpy as np 
mu005, sigma005 = 0, np.std(data_nh_std)*0.05
noise005 = np.random.normal(mu005, sigma005, data_nh_std.shape)
noise005[:,12:17]=0                        # label不可以加噪声

mu010, sigma010 = 0, np.std(data_nh_std)*0.1
noise010 = np.random.normal(mu010, sigma010, data_nh_std.shape)
noise010[:,12:17]=0

mu020, sigma020 = 0, np.std(data_nh_std)*0.2
noise020 = np.random.normal(mu020, sigma020, data_nh_std.shape)
noise020[:,12:17]=0

mu030, sigma030 = 0, np.std(data_nh_std)*0.3
noise030 = np.random.normal(mu030, sigma030, data_nh_std.shape)
noise030[:,12:17]=0

mu050, sigma050 = 0, np.std(data_nh_std)*0.5
noise050 = np.random.normal(mu050, sigma050, data_nh_std.shape)
noise050[:,12:17]=0

data_nh_std_005=data_nh_std+noise005
data_nh_std_010=data_nh_std+noise010
data_nh_std_020=data_nh_std+noise020
data_nh_std_030=data_nh_std+noise030
data_nh_std_050=data_nh_std+noise050

train005, test005,valid005, train_X005,train_y005, test_X005,test_y005, valid_X005,valid_y005 =data_processing(data=data_nh_std_005)

MLP_trainX005, MLP_validX005, MLP_testX005 = datalining(train_X005,
                                                        valid_X005,
                                                        test_X005
                                                       )

train010, test010,valid010, train_X010,train_y010, test_X010,test_y010, valid_X010,valid_y010 =data_processing(data=data_nh_std_010)

MLP_trainX010, MLP_validX010, MLP_testX010 = datalining(train_X010,
                                                        valid_X010,
                                                        test_X010
                                                       )

train020, test020,valid020, train_X020,train_y020, test_X020,test_y020, valid_X020,valid_y020 =data_processing(data=data_nh_std_020)

MLP_trainX020, MLP_validX020, MLP_testX020 = datalining(train_X020,
                                                        valid_X020,
                                                        test_X020
                                                       )

train030, test030,valid030, train_X030,train_y030, test_X030,test_y030, valid_X030,valid_y030 =data_processing(data=data_nh_std_030)

MLP_trainX030, MLP_validX030, MLP_testX030 = datalining(train_X030,
                                                        valid_X030,
                                                        test_X030
                                                       )

train050, test050,valid050, train_X050,train_y050, test_X050,test_y050, valid_X050,valid_y050 =data_processing(data=data_nh_std_050)

MLP_trainX050, MLP_validX050, MLP_testX050 = datalining(train_X050,
                                                        valid_X050,
                                                        test_X050
                                                       )


# ## GRU

# In[67]:


train_x_tensor005,train_y_tensor,valid_x_tensor,valid_y_tensor,test_x_tensor,test_y_tensor=data2tensor(MLP_trainX=MLP_trainX005,
                                                                                                    train_y=train_y,
                                                                                                    MLP_validX=MLP_validX,
                                                                                                    valid_y=valid_y,
                                                                                                    MLP_testX=MLP_testX,
                                                                                                    test_y=test_y
                                                                                                    )


# In[68]:


train_x_tensor005,train_y_tensor,valid_x_tensor010,valid_y_tensor,test_x_tensor010,test_y_tensor=data2tensor(MLP_trainX=MLP_trainX005,
                                                                                                            train_y=train_y,
                                                                                                            MLP_validX=MLP_validX010,
                                                                                                            valid_y=valid_y,
                                                                                                            MLP_testX=MLP_testX010,
                                                                                                            test_y=test_y
                                                                                                            )


# In[69]:


train_x_tensor005,train_y_tensor,valid_x_tensor020,valid_y_tensor,test_x_tensor020,test_y_tensor=data2tensor(MLP_trainX=MLP_trainX005,
                                                                                                            train_y=train_y,
                                                                                                            MLP_validX=MLP_validX020,
                                                                                                            valid_y=valid_y,
                                                                                                            MLP_testX=MLP_testX020,
                                                                                                            test_y=test_y
                                                                                                            )


# In[70]:


train_x_tensor005,train_y_tensor,valid_x_tensor030,valid_y_tensor,test_x_tensor030,test_y_tensor=data2tensor(MLP_trainX=MLP_trainX005,
                                                                                                            train_y=train_y,
                                                                                                            MLP_validX=MLP_validX030,
                                                                                                            valid_y=valid_y,
                                                                                                            MLP_testX=MLP_testX030,
                                                                                                            test_y=test_y
                                                                                                            )


# In[71]:


train_x_tensor005,train_y_tensor,valid_x_tensor050,valid_y_tensor,test_x_tensor050,test_y_tensor=data2tensor(MLP_trainX=MLP_trainX005,
                                                                                                            train_y=train_y,
                                                                                                            MLP_validX=MLP_validX050,
                                                                                                            valid_y=valid_y,
                                                                                                            MLP_testX=MLP_testX050,
                                                                                                            test_y=test_y
                                                                                                            )


# In[72]:


RNN_fit('GRU',
        epoches=100,
        LR=0.005,
        train_x_tensor=train_x_tensor005,
        train_y_tensor=train_y_tensor,
        valid_x_tensor=valid_x_tensor,#010 020 030 050
        valid_y_tensor=valid_y_tensor,
        test_x_tensor=test_x_tensor,#010 020 030 050
        test_y_tensor=test_y_tensor,
        )


# In[ ]:




