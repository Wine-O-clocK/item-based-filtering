import pandas as pd
import numpy as np
import numpy as np

#데이터 준비
rating_data = pd.read_csv('./ratings.csv')
wine_data = pd.read_csv('./wine_list.csv')

# rating_data.head()

# wine_data.head()

#필요없는 컬럼삭제 (axis=1(좌우))
# rating_data.drop('컬럼명 이름을 적어주세요', axis=1, inplace=True)
wine_data.drop('wineImage', axis=1, inplace=True)
wine_data.drop('wineNameEng', axis=1, inplace=True)

wine_data.drop('wineType', axis=1, inplace=True)
wine_data.drop('winePrice', axis=1, inplace=True)
wine_data.drop('wineSweet', axis=1, inplace=True)
wine_data.drop('wineBody', axis=1, inplace=True)
wine_data.drop('wineVariety', axis=1, inplace=True)
wine_data.drop('aroma1', axis=1, inplace=True)
wine_data.drop('aroma2', axis=1, inplace=True)
wine_data.drop('aroma3', axis=1, inplace=True)

# wine_data.head()

#와인데이터와 평점데이터를 머지
#merge : 고유값(key)을 기준으로 병합. on을 기준으로 병합. 
user_wine_data = pd.merge(rating_data, wine_data, on="wineName")

# 피봇테이블 생성
# 와인-사용자 피봇테이블 (index=wineId, column:userId)
wine_user_pivot = user_wine_data.pivot_table('rating', index='wineName', columns='userId')

# wine_user_pivot.head()
# print(wine_user_pivot)

#평점을 매기지 않은 데이터는 NaN으로 나옴 -> 0으로 처리
wine_user_pivot.fillna(0,inplace=True)
wine_user_pivot.head()
# print(wine_user_pivot)

# user_wine_data.shape
# print(wine_user_pivot)
# """## 아이템기반 협업 필터링
# - 유사한 아이템끼리 추천해주는 방식
# - 평점이 비슷한 아이템(와인) 추천
# - 코사인 유도값을 이용해 유사도 계산
# """

from sklearn.metrics.pairwise import cosine_similarity

similarity_rate = cosine_similarity(wine_user_pivot)
print(similarity_rate)


#유사도 값을 가진 데이터프레임 생성
# -> 각 아이팀끼리 서로 유사한 정보의 값을 가지게 됨
# -> 유사도가 가까운 영화일수록 1에 가깝고, 자기자신과 같은 영화일때 유사도 값 1
similarity_rate_df = pd.DataFrame(
    data=similarity_rate,
    index=wine_user_pivot.index,
    columns=wine_user_pivot.index
)

similarity_rate_df.head()

#가장 유사도가 높은 top5
def recommand_wine(wineName):
  # 맨 첫번째는 자기 자신이라 제외
  return similarity_rate_df[wineName].sort_values(ascending=False)[1:6]

recommand_wine('플립 플랍 샤도네이')