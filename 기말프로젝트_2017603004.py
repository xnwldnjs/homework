#!/usr/bin/env python
# coding: utf-8

# # 기말 프로젝트 -  kaggle 'Bike Sharing and Demand'

# # 2017603004 이지원

# In[1]:


#필요한 모듈 import하기


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[3]:


#데이터셋 불러오기


# In[4]:


train = pd.read_csv("bike_train.csv",parse_dates=["datetime"]) # bike_train.csv읽어와서 train에 저장 (datetime은 날짜로 해석하기위해 parse_dates)
train.shape #데이터 행렬 사이즈 출력


# In[5]:


train.head() #train 상위 5개 확인


# In[6]:


test=pd.read_csv("bike_test.csv",parse_dates=["datetime"])# bike_test.csv읽어와서 test에 저장 (datetime은 날짜로 해석하기위해 parse_dates)
test.shape#데이터 행렬 사이즈 출력


# In[7]:


test.head() #test 상위 5개 확인


# In[8]:


train.info() #train의 정보 확인


# In[9]:


test.info() #test의 정보 확인


# - train에 있는 변수인 casual, registered, count 가 test에는 없음
# - casual+registered=count임을 알 수 있음

# # 데이터 전처리

# In[10]:


# train에 연, 월, 일, 시, 분, 초 나타내는 column 생성
train["year"] = train["datetime"].dt.year # 연을 나타내는 column 이름은 year
train["month"] = train["datetime"].dt.month # 월을 나타내는 column 이름은 month
train["day"] = train["datetime"].dt.day # 일을 나타내는 column 이름은 day
train["hour"] = train["datetime"].dt.hour # 시를 나타내는 column 이름은 hour
train["minute"] = train["datetime"].dt.minute # 분을 나타내는 column 이름은 minute
train["second"] = train["datetime"].dt.second # 초를 나타내는 column 이름은 second

train["dayofweek"] = train["datetime"].dt.dayofweek # 요일 (0~6 은 각 월~일)

train.shape # train의 행렬 사이즈 출력


# In[11]:


train[["datetime", "year", "month", "day", "hour", "minute", "second", "dayofweek"]].head() # train데이터 중 정한 8개("datetime","year",등)의 column 출력


# In[12]:


#숫자로 되어있는 dayofweek을 요일로 바꿔줌 (0~6을 각 월~일 로)
train.loc[train["dayofweek"]==0, "dayofweek(string)"]= "Mon" 
train.loc[train["dayofweek"]==1, "dayofweek(string)"]= "Tue"
train.loc[train["dayofweek"]==2, "dayofweek(string)"]= "Wed"
train.loc[train["dayofweek"]==3, "dayofweek(string)"]= "Thu"
train.loc[train["dayofweek"]==4, "dayofweek(string)"]= "Fri"
train.loc[train["dayofweek"]==5, "dayofweek(string)"]= "Sat"
train.loc[train["dayofweek"]==6, "dayofweek(string)"]= "Sun"

train.shape #train의 행렬 사이즈 출력 


# In[13]:


train[["datetime", "dayofweek", "dayofweek(string)"]].head() #train데이터 중 정한 3개("datetime","dayofweek","dayofweek(string)")의 column 출력(상위 다섯개)


# In[14]:


# test에 연, 월, 일, 시, 분, 초 나타내는 column 생성
test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["second"] = test["datetime"].dt.second

test["dayofweek"] = test["datetime"].dt.dayofweek

test.shape # test의 행렬 사이즈 출력


# In[15]:


test[["datetime", "year", "month", "day", "hour", "minute", "second", "dayofweek"]].head() # test데이터 중 정한 8개("datetime","year",등)의 column 출력(상위 다섯개)


# In[16]:


#숫자로 되어있는 dayofweek을 요일로 바꿔주기 (0~6을 각 월~일 로)
test.loc[test["dayofweek"] == 0, "dayofweek(string)"] = "Mon"
test.loc[test["dayofweek"] == 1, "dayofweek(string)"] = "Tue"
test.loc[test["dayofweek"] == 2, "dayofweek(string)"] = "Wed"
test.loc[test["dayofweek"] == 3, "dayofweek(string)"] = "Thu"
test.loc[test["dayofweek"] == 4, "dayofweek(string)"] = "Fri"
test.loc[test["dayofweek"] == 5, "dayofweek(string)"] = "Sat"
test.loc[test["dayofweek"] == 6, "dayofweek(string)"]= "Sun"
test.shape


# In[17]:


test[["datetime", "dayofweek", "dayofweek(string)"]].head() # test데이터 중 정한 3개("datetime","dayofweek","dayofweek(string)")의 column 출력


# # 데이터 시각화하기

# In[18]:


#연,월,일,시,분,초 별로 train데이터의 자전거 대여량을 막대그래프로 나타내기 ( 자전거 대여량에 영향을 미칠만한 feature 찾기)

f, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,10))

sb.barplot(data=train, x="year", y="count", ax=ax[0][0])
ax[0][0].set_title("Rent count - Year")
sb.barplot(data=train, x="month", y="count", ax=ax[0][1])
ax[0][1].set_title("Rent count - Month")
sb.barplot(data=train, x="day", y="count", ax=ax[0][2])
ax[0][2].set_title("Rent count-day")
sb.barplot(data=train, x="hour", y="count", ax=ax[1][0])
ax[1][0].set_title("Rent count - hour")
sb.barplot(data=train, x="minute", y="count", ax=ax[1][1])
ax[1][1].set_title("Rent count - minute")
sb.barplot(data=train, x="second", y="count", ax=ax[1][2])
ax[1][2].set_title("Rent count - second")

plt.show()


# # 위 그래프를 통해 알 수 있는 것:
# 
# - 연: 2011년보다 2012년에 대여량 증가
# - 월: 12월 ~ 2월 보다 6 ~ 8월에 자전거를 더 많이빌림 즉, 겨울<여름
# - 일: train데이터에는 일수가 1일부터 19일까지만 존재 (나머지는 test데이터에 존재) 즉, day column을 feature에서 빼야함 (과적합이 일어날 수 있음)
# - 시간: 새벽에는 자전거를 거의 빌리지 않고 출퇴근 시간대에 많이빌림
# - 분, 초 : 0 으로 의미없음 즉, feature에서 빼야함
# 

# # 연, 월

# In[19]:


# 연도와 월 컬럼을 조합 (yyyy-m 형식)
train["year-month"] = train["year"].astype('str') + "-" + train["month"].astype('str')
train[["datetime", "year-month"]].head()


# In[20]:


# 연도별 대여량, 월별 대여량, 그리고 연-월 별 대여량을 막대그래프로 시각화 (의미있는 데이터인지 확인)
f, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
plt.subplots_adjust(hspace = 0.5)
sb.barplot(data=train, x="year", y="count", ax=ax[0])
ax[0].set_title("Rent count - Year")
sb.barplot(data=train, x="month", y="count", ax=ax[1])
ax[1].set_title("Rent count - Month")
sb.barplot(data=train, x="year-month", y="count", ax=ax[2])
ax[2].set_title("Rent count - Year_Month")

plt.show()


# # 알 수 있는 것:
# - year-month 그래프에서 보면 2011년보다 2012년의 대여량이 많은데, 그래프의 모양은 비슷한걸로 보아 2011년보다 2012년에 회사의 성장으로 인해 대여량이 늘어난 것으로 보임 (그렇게 의미있는 데이터는 아님)
# 

# # 시간, 근무일, 요일

# In[21]:


# 시간, 근무일, 요일 별 대여량을 꺾은선 그래프로 시각화
f, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
plt.subplots_adjust(hspace = 0.5)
sb.pointplot(data=train, x="hour", y="count", ax=ax[0])
ax[0].set_title("Rent count - hour")
sb.pointplot(data=train, x="hour", y="count", hue="workingday", ax=ax[1])
ax[1].set_title("Rent count - hour (workingday)")
sb.pointplot(data=train, x="hour", y="count", hue="dayofweek(string)", ax=ax[2])
ax[2].set_title("Rent count - hour (dayofweek)")
plt.show()


# # 알 수 있는 것
# 
# - 아까 위에서 시간에 따른 대여량 그래프를 봤을때 출퇴근시간에 대여량이 많은 것을 확인할 수 있었음
# 
# - 시간-대여량 그래프에 근무일을 기준으로 대여량을 나눈 것이 두번째 그래프인데, 근무일에는 출, 퇴근 시간에 대여량이 많은 반면, 쉬는날에는 출,퇴근 시간 상관 없이 활동하기 좋은 오후시간에 대여량이 많은 것을 확인할 수 있음
# 
# - 시간-대여량 그래프에 요일을 기준으로 대여량을 나눈 것이 세번째 그래프인데, 표현하게 되면 맨 밑의 그래프처럼 나타나게 되는데, 월~금에는 출퇴근 시간대의 대여량이 많고 쉬는 날인 토~일에는 오후 시간대에 대여량이 많음 

# # 대여량 (log transformation)

# In[22]:


# 최종적으로 예측해야 할 Label인 대여량 column 시각화
count_graph=sb.distplot(train["count"])


# - 분포가 너무 크게 나타남

# In[23]:


# 분포를 줄이기 위해 Log Transformation를 취한 뒤 시각화
train["count_log"] = np.log(train["count"] + 1)
count_log_graph=sb.distplot(train["count_log"])


# # feature, label 설정

# In[24]:


#정답을 맞히는데 도움이 되는 값들
feature_list = ["season","holiday", "workingday","weather", "temp", "atemp", "humidity", "windspeed", "year", "hour","dayofweek"]
feature_list


# In[25]:


#맞혀야 하는 값
label_list=["count_log"]
label_list


# # train, test 데이터  분리

# In[26]:


X_train = train[feature_list] 
X_train.head() # X_train 상위 다섯개


# In[27]:


X_test = test[feature_list]
X_test.head() # X_test 상위 다섯개


# In[28]:


y_train = train[label_list]
y_train.head() # y_train 상위 다섯개


# # XGBoost 적용

# In[29]:


import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=500,max_depth = 6,learning_rate=0.02,seed=37) #파라미터 계속 바꿔봄


# In[30]:


model.fit(X_train,y_train) # 학습


# In[31]:


log_predictions=model.predict(X_test) #예측


# In[32]:


log_predictions.shape #데이터 사이즈 출력


# In[33]:


log_predictions #변수 출력


# In[34]:


predictions=np.exp(log_predictions)-1 #예측 값은 Log transformation된 값이므로 원상 복구해주기
predictions.shape # prediction의 사이즈 출력


# In[35]:


predictions #변수 출력


# In[36]:


submission=pd.read_csv("sampleSubmission.csv ") #제출하기위해 내려받은 파일 중 sampleSubmission.csv불러오기
submission.shape


# In[37]:


submission.head() #상위 다섯개 확인


# In[38]:


submission["count"]=predictions #count 자리에 prediction값들 넣기
submission.shape


# In[39]:


submission.head() #상위 다섯개 확인


# In[40]:


submission.to_csv("sampleSubmission2.csv", index=False) # sampleSubmission2..csv라는 이름으로 저장 (이 파일을 kaggle에 제출)


# In[ ]:




