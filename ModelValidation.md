# Kaggle-Tutorial

## Intro to Machine Learning

---

## 모델  검증Validation

원본: [Your First Machine Learning Model](https://www.kaggle.com/dansbecker/model-validation)

당신의 모델을 만들어 봤었다. 정말 좋지 않나요?

이번 과정에서는, 모델 검증에 대해 배움으로써, 모델의 품질을 측정할 수 있게 될 것입니다. 모델품질을 측정하는 것은 모델을 계속해서 개선해나감의 핵심이다.

### 모델Model 검증Validation 이란?

당신은 당신이 지금까지 만든 거의 모든 모델에 대해 평가를 하길 원할 것이다. 거의 대부분(전부는 아니지만)의 어플리케이션에서 모델 품질의 관련 척도는 예측 정확도이다. 다른말로, 모델의 예측이 실제와 가깝게됨이다.

많은 이들이 예측정확도를 측정할 때 실수를 하곤한다. 그들른 훈련 데이터오 예측을 하고 그 예측을 훈련 데이터의 목표값과 비교한다. 이 방식의 문제점과 해결방법은 곧 보게 될 것이나, 우리는 먼저 이를 어떻게 다루어야 할지 생각해보자.

먼저 모델 품질에 대해 이해가능한 방식으로 요약하는 것이 첫번째로 필요할 것이다. 만약 당신이 10,000 채의 집값에 대한 예측값과 실제값을 비교해본다면, 좋은 예측값과 나쁜예측 값이 섞여 있는 것을 볼 것이다. 10,000개의 예측과 실제 값의 리스트를 살펴보는 것은 무의미할 것이다. 우리는 이를 하나의 측정항목metric으로 요약할 필요가 있다.

모델 품질을 요약하는 많은 측정항목metric이 있지만, 우리는 평균절대오차(Mean Absolute Error, MAE라고도 함)에서 부터 시작하겠다. 마지맏 단어인 Error에서 부터 시작하여 이 metric에 대해 분석해보자.

각 주택의 예측 오차는 다음과 같다.

```
오차error = 실제값 - 예측값
```

때문에, 만약 집한채의 가격이 150,000\$ 이고 당신이 예측한 값이 100,000\$이면 오차는 50,000\$이다.

MAE metric을 이용하면, 우리는 각 오차들의 절대값을 얻는다. 이는 각 오차를 양수로 변환한다. 그런다음 이 오차의 절대값에 대한 평균을 구한다. 이것이 우리 모델의 척도이다. 쉬운말로 이렇게 표현할 수도 있다.

> 평균적으로, 우리 예측값은 X정도 벗어났다.

MAE를 계산하려면, 우리는 먼저 모델이 필요하다.

In [1]:    

```python
# Data Loading Code Hidden Here

import pandas as pd

# Load data

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows with missing price values

filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features

y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor

# Define model

melbourne_model = DecisionTreeRegressor()

# Fit model

melbourne_model.fit(X, y)
```

Out [1]

> DecisionTreeRegressor()

우리가 한 개의 모델에 대해, 평균절대오차를 구하는 방법은 다음과 같다.
<br/><br/>


In [2]:

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

Out [2]:

> 434.71594577146544

### 표본내"In-Sample" 점수 문제

방금 계산한 척도를 "표본 내" 점수라 부를 수 있다. 우리는 모델을 만들고 평가하기 위해 주택들을 하나의 표본Sample으로 사용했다. 이것이 나쁜 이유는 다음과 같다.

다음을 가정해 보자, 대형 부동산 시장에서, 문의 색깔은 집값과는 상관관계가 없다.

그러나, 모델을 만듦에 있어 사용된 표본 데이터들에 있어, 녹색 문의 집들은 대부분 비싸게 책정되었다.
이 모델의 일은 집값을 예측하는 패턴을 찾아내는 것인데, 모델은 이러한 패턴을 찾게 되었고, 이에 따라 초록색 문이 있는 주택은 항상 높은 가격으로 예측될 것이다.

이 패턴은 훈련데이터에서 파생된 것derived from으로, 훈련데이터 상에서는 정확하게 나타날 것이다.

하지만, 이 모델이 새로운 데이터를 보게될 때, 이 패턴이 유지되지 않는다면, 실제 사용에서 모델은 매우 부정확할 것이다.

모델의 실제 가치는 새로운 데이터에 대한 예측을 통해서 확보되기에, 모델을 구축하는데 사용되지 않은 데이터를 이용하여 성능을 측정한다. 이를 위한 가장 확실한 방법은 일부데이터를 모델 구축 프로세스레서 제외시킨 다음, 이 제외되어 모델이 한 번도 보지 못한 데이터를 사용하여 모델의 정확도를 테스트하는 것이다. 이것을 검증데이터Validaion data 라고 한다.



### 코딩해보자

skit-learn 라이브러리에는 데이터를 두 조각으로 나누는 train_test_split 함수가 있다. 이를 사용하여, 데이터 중 일부는 모형에 적합한 교육 데이터로 사용하고 다른 데이터는 검증 데이터로 사용하여mean_absolute_error를 계산할 것이다.

코드는 다음과 같다.

```python
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```

> 258930.03550677857

### Wow!

표본 내 데이터에 대한 당신의 평균절대오차는 약 500달러였고, 샘플 외에서는 25만 달러이상이다.
이것이 거의 정확하게 맞는 모델과 실용적으로의 거의 쓸 수 없는 모델의 차이이다. 참고로 검증 데이터의 평균 집값은 110만 달러이다. 따라서 새로운 데이터에 따른 오차는 평균적 집값의 1/4수준이다.

### 당신의 차례

이 모델을 개선하기 전에 먼저 **[Model Validation](exercise-your-first-machine-learning-model.ipynb)**을 스스로 해보자.


