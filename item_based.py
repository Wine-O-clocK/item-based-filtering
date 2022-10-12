# -*- coding: utf-8 -*-
"""아이템기반_협업_필터링.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17-W4DS_fWE7_oWBfd8uUu2BPoWjTaBDT

##### ✅ ratings.csv -> user가 와인에 평가를 매긴 데이터
- userId
- wineId
- rating
  - 검색하기 (1점)
  - 클릭하기 (2점)
  - 저장하기 (3점)
  - 평가하기 (4점)
    - '그냥그래요' 선택 시 0점 기록 
    - ‘별로에요’ 선택 시 -1점 기록


##### ✅ wines.csv -> 와인정보 데이터
- wineImage
- wineName
- wineNameEng
- wineType
- winePrice
- wineSweet
- wineBody
- wineVariety
- aroma1,aroma2,aroma3

### 협업 필터링을 위한 작업
"""

import pandas as pd
import numpy as np
import numpy as np

#데이터 준비
rating_data = pd.read_csv('./ratings.csv')
wine_data = pd.read_csv('./wine_list.csv')

rating_data.head()

wine_data.head()

#필요없는 컬럼삭제 (axis=1(좌우))
#rating_data.drop('컬럼명 이름을 적어주세요', axis=1, inplace=True)
wine_data.drop('삭제할 컬럼명', axis=1, inplace=True)

wine_data.head()

#와인데이터와 평점데이터를 머지
#merge : 고유값(key)을 기준으로 병합. on을 기준으로 병합. 
user_wine_data = pd.merge(rating_data, wine_data, on='병합할 컬럼명')

user_wine_data.head()

# 피봇테이블 생성
# 와인-사용자 피봇테이블 (index=wineId, column:userId)
wine_user_pivot = user_wine_data.pivot_table('rating', index='wineName', columns='userId')

wine_user_pivot.head()

#평점을 매기지 않은 데이터는 NaN으로 나옴 -> 0으로 처리
wine_user_pivot = wine_user_pivot.fillna(0)
wine_user_pivot.head()

wine_user_pivot.shape

"""## 아이템기반 협업 필터링
- 유사한 아이템끼리 추천해주는 방식
- 평점이 비슷한 아이템(와인) 추천
- 코사인 유도값을 이용해 유사도 계산
"""

from sklearn.metrics.pairwise import cosine_similarity

similarity_rate = cosine_similarity(wine_user_pivot)
print(similarity_rate)

#유사도 값을 가진 데이터프레임 생성
# -> 각 아이팀끼리 서로 유사한 정보의 값을 가지게 됨
# -> 유사도가 가까운 영화일수록 1에 가깝고, 자기자신과 같은 영화일때 유사도 값 1
similarity_rate_df = pd.DataFrame(
    data = similarity_rate,
    index = wine_user_pivot.index,
    columns = wine_user_pivot.index
)

similarity_rate_df.head()

#가장 유사도가 높은 top5
def recommand_wine(wineName):
  # 맨 첫번째는 자기 자신이라 제외
  return similarity_rate_df[wineName].sort_values(ascending=False)[1:6]

recommand_wine('와인이름을 검색하세요')