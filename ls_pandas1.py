import pandas as pd
import numpy as np
url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)
print(mydata.head())

mydata.shape

mydata['gender'].head()
mydata['gender'].tail()


print(mydata[['gender', 'midterm']].head())

print(mydata[mydata['midterm'] > 15].head())

#iloc - indexing 숫자로 인덱싱 했을 떄 사용함.
mydata.iloc[:,1]
mydata.iloc[:,[1]]
mydata.iloc[:,[1, 0 , 1]]
mydata.head()
mydata.iloc[1:5,2]

mydata.iloc[:,[1]].squeeze() #DataFrame에서 Series로 하나만 가져와서 변환 할떄 근데 하나가져오면 자동변환 요즘되는걸로 알고있음

mydata.loc[:,"midterm"]
mydata.loc[1:4,"midterm"]

mydata.head()
mydata.columns.unique() # 
mydata.nunique()

mydata.loc[mydata['midterm'] <= 15, ['student_id', 'final']].head()

mydata[mydata['midterm'].isin([28, 38, 52])].head()

#중간고사 점수 28 38 52 인 애들의 기말고사 점수와 성별 정보를 가져오세요/
#loc만 True, False 인덱싱만 가능함
mydata.loc[mydata['midterm'].isin([28, 38, 52]), ['final', 'gender']]


# 먼저 조건 필터링 → 그 결과에서 위치로 추출
filtered = mydata[mydata['midterm'].isin([28, 38, 52])]
result = filtered.iloc[:, [mydata.columns.get_loc('final'), mydata.columns.get_loc('gender')]] #np.where도 가능함

mydata.iloc[0,1] =np.nan
mydata.iloc[4,0] = np.nan

mydata.head()

mydata['gender'].isna().sum()

mydata.dropna()

#1번
mydata['student_id'].isna().head()
#2번
~mydata['student_id'].isna().head()

#3번
mydata['gender'].isna().head()

mydata['student_id'].isna() & ~mydata['student_id'].isna()

mydata['total'] = mydata['midterm'] + mydata['final']
mydata.head(3)

mydata['average'] = (mydata['total'] /2).rename('avg',inplace=True) 
mydata.head(3)

help(pd.Series.rename)

#concat

import pandas as pd
# 첫 번째 데이터 프레임 생성
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})

df2 = pd.DataFrame({
'A': ['A3', 'A4', 'A5'],
'B': ['B3', 'B4', 'B5']
})
result = pd.concat([df1, df2]) # pd.concat([df1, df2], ignore_index=True)
result #나중에 인덱스를 초기화 하고 싶을떄  result.reset_index(drop=True)

df3 = pd.DataFrame({
'C': ['C0', 'C1', 'C2'],
'D': ['D0', 'D1', 'D2']
})
result = pd.concat([df1, df3], axis=1)
print(result)


df4 = pd.DataFrame({
'A': ['A2', 'A3', 'A4'],
'B': ['B2', 'B3', 'B4'],
'C': ['C2', 'C3', 'C4']
})

result = pd.concat([df1, df4], join='inner')