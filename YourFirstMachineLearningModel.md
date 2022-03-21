# Kaggle-Tutorial

## Intro to Machine Learning

---

### 모델링을 위한 데이터 선택

원본: [Your First Machine Learning Model](https://www.kaggle.com/dansbecker/your-first-machine-learning-model)

당신의 데이터셋은 너무 많은 변수들을 가지고 있어, 머리로 정리하거나 그럴듯하게 표현할 방법이 없다. 이 엄청난 데이터양을 이해할 수 있는 수준으로 줄이는 방법은 무엇일까?

우리는 직관적으로 몇몇개의 변수를 고르는 것으로 시작하자. 이후 과정에서는 당신은 변수의 우선 순위를 정하는 통계 기법을 보게될 것이다.

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.columns
```

```
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 'Regionname', 'Propertycount'], dtype='object')
```

<br/>

```
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
```

데이터에서 subset하위집합을 선택함은 여러 방법이 있다. [Pandas course](https://www.kaggle.com/learn/pandas)d에서 더 자세히 다루겠지만, 지금 우리는 두가지 접근법에 초점을 맞추겠다.

1. 점-표기법: "예측 목표값"을 선택하는데 사용
2. column list에서 선택: "Features특징들"을 선택하는데 사용

### 예측 목표 선택

**dot-notation**점-표기법으로 변수를 꺼낼 수 있다. 이 단일 열은 **Series**에 저장되며, 이는 열이 하나뿐인 DataFrame이라 봐도 무방하다.

점 표기법을 사용하여 예측 대상이라고 하는 열을 선택할 것이다. 일반적으로 예측 대상을 y라 한다. 멜버른 데이터에 집값을 저장하기 위해 필요한 코드는

```
y = melbourne_data.Price
```

### "Features특징들" 선택하기

모델(이후 예측에 사용)에 입력되는 열들을 "Features"라 한다. 우리의 경우 집값을 결정하는 column들이 될 것이다. 때론 target값을 제외한 모든 열이 features로 사용되곤 한다. 때론 더 적은 features를 사용하는 것이 더 나을 수도 있다.

지금은 몇 가지 기능만 갖춘 모델을 제작하겠다. 추후 서로 다른 features로 제작된 모델들을 반복적으로 만들고 이를 비교하는 방법에 대해 알아볼 것이다.

대괄호 안에 열 이름 목록을 제공하여 여러 features를 선택한다. 목록의 각 항목은 따옴표가 있는 문자열이어야 한다. 예를 들어

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
```

<br/>

관례상 이 데이터를 X라 부른다.

```
X = melbourne_data[melbourne_features]
```

우리가 집값을 예측하기 위해 사용할 데이터를 빠르게 보기 검토하기 위해 `describe`메서드와 `head`메서드를 사용하여 상위 몇개의 row행들을 사용해보자.

```
X.describe()
```

|       | Rooms       | Bathroom    | Landsize     | Lattitude   | Longtitude  |
|-------|-------------|-------------|--------------|-------------|-------------|
| count | 6196.000000 | 6196.000000 | 6196.000000  | 6196.000000 | 6196.000000 |
| mean  | 2.931407    | 1.576340    | 471.006940   | -37.807904  | 144.990201  |
| std   | 0.971079    | 0.711362    | 897.449881   | 0.075850    | 0.099165    |
| min   | 1.000000    | 1.000000    | 0.000000     | -38.164920  | 144.542370  |
| 25%   | 2.000000    | 1.000000    | 152.000000   | -37.855438  | 144.926198  |
| 50%   | 3.000000    | 1.000000    | 373.000000   | -37.802250  | 144.995800  |
| 75%   | 4.000000    | 2.000000    | 628.000000   | -37.758200  | 145.052700  |
| max   | 8.000000    | 8.000000    | 37000.000000 | -37.457090  | 145.526350  |

```
X.head()
```

|   | Rooms | Bathroom | Landsize | Lattitude | Longtitude |
|---|-------|----------|----------|-----------|------------|
| 1 | 2     | 1.0      | 156.0    | -37.8079  | 144.9934   |
| 2 | 3     | 2.0      | 134.0    | -37.8093  | 144.9944   |
| 4 | 4     | 1.0      | 120.0    | -37.8072  | 144.9941   |
| 6 | 3     | 2.0      | 245.0    | -37.8024  | 144.9993   |
| 7 | 2     | 1.0      | 256.0    | -37.8060  | 144.9954   |

이러한 명령어들로 당신의 데이터를 시각적으로 확인하함은 데이터사이언티스트에게 중요한 부분이다. 당신은 자주 추후 검토할만한 데이터셋안에서 놀라움을 발견할 수 있을 것이다.

---

### 당신의 모델을 빌드하기

**scikit-learn** 라이브러리를 사용하여 당신의 모델을 만들어 볼 것이다. 코딩시, 이 라이브러리는 아래 샘플 코드와 같이 **scikit-learn**이라 적는다. scikit-learn은 특별히DataFraemes에 저장된 데이터 유형을 모델링함에 있어 쉽고 매우 대중적인 라이브러리이다.

모델을 구축하는 단계는 다음 단계들은:

* 정의: 어떤 유형의 모델인가? 의사 경절 트리? 다른 종류의 모델? 모델 유형의 다른 파라미터들도 지정된다.
* fit: 제공된 데이터에서 패턴을 캡쳐한다. 모델링의 중심.
* 예측: 말그대로
* 평가: 모델의 예측이 얼마나 정확한지 확인한다.

다음은 skit-learn으로 의사 결정 트리 모델을 정의하고 특징 및 대상 변수에 적합시키는 예다.

```
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)
```

`DecisionTreeRegressor(random_state=1)`

<br/>

많은 머신러닝의 모델들은 어느정도의 무작위성을 허용한다. `random_state`에 숫자를 지정하면 각 실행에서 동일한 결과를 얻을 수 있다. 이건 좋은 관행으로 고려할 수 있다.
당신이 정한 숫자를 사용함이 모델의 품질에는 유의미한 영향을 주지는 않는다.

이에 우리는 예측에 사용할 피팅된 모델을 가지게 되었다.

실제로, 당신이 예측하길 원하는 것은 이미 가격을 책정한 집들이 아니라 시장에 나올 새로운 딪에 대한 예측일 것이다. 그러나 우리는 예측 기능이 어떻게 작동하는지 보기 위해 훈련 데이터의 처음 몇 행에 대해 예측할 것이다.

```
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```

```
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
```

### 당신의 차례

**[Model Building Exercise](exercise-your-first-machine-learning-model.ipynb)**

