# Kaggle-Tutorial
## Intro to Machine Learning
***
## 기본 데이터 탐구
당신의 데이터터를 load하고 이해하자.

원본: [Basic Data Exploration](https://www.kaggle.com/dansbecker/basic-data-exploration)

### Pandas을 사용하여 데이터에 익숙해지기

머신러닝 프로젝트의 첫 단계는 해당 데이터에 친숙해지는 것이다. 
이를 위해서 Pandas 라이브러리를 사용할 것이다
Pandas는 과학자들이 데이터를 탐색하고 조작하기 위해 사용하는 주요 도구다. 
대부분의 사람들은 코드에서 Pandas를 pd로 줄인다. 우리는 이 명령을 이렇게 사용한다.

```python
import pandas as pd
```

Pandas 라이브러리의 가장 중요한 부분은 DataFramedl다. 
DataFrame은 표로 생각할 수 있는 유형의 데이터를 보관한다. 
이는 Excel의 시트 또는 SQL 데이터베이스의 테이블과 유사하다.

Pandas는 이러한 타입의 데이터로 할 수 있는 대부분을 위한 강력한 메서드들을 가진다.

예를 들어 , 우리는 호주 멜머른의 [집값에 대한 데이터](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot)를 살펴볼 것이다.
실습에서는 아이오와의 집값에 대한 새 데이터셋에, 동일한 프로세스를 적용할 수 있다.

```python
# 빠른 접근을 위해 파일경로를 변수로 저장
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# 데이터를 읽고 DataFrame에 데이터를 적재하여 melbourne_data로 명명
melbourne_data = pd.read_csv(melbourne_file_path) 
# Melbourne data의 데이터 요약 출력
melbourne_data.describe()
```

~~~
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rooms</th>
      <th>Price</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13580.000000</td>
      <td>1.358000e+04</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13518.000000</td>
      <td>13580.000000</td>
      <td>7130.000000</td>
      <td>8205.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
      <td>13580.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.937997</td>
      <td>1.075684e+06</td>
      <td>10.137776</td>
      <td>3105.301915</td>
      <td>2.914728</td>
      <td>1.534242</td>
      <td>1.610075</td>
      <td>558.416127</td>
      <td>151.967650</td>
      <td>1964.684217</td>
      <td>-37.809203</td>
      <td>144.995216</td>
      <td>7454.417378</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.955748</td>
      <td>6.393107e+05</td>
      <td>5.868725</td>
      <td>90.676964</td>
      <td>0.965921</td>
      <td>0.691712</td>
      <td>0.962634</td>
      <td>3990.669241</td>
      <td>541.014538</td>
      <td>37.273762</td>
      <td>0.079260</td>
      <td>0.103916</td>
      <td>4378.581772</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>8.500000e+04</td>
      <td>0.000000</td>
      <td>3000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1196.000000</td>
      <td>-38.182550</td>
      <td>144.431810</td>
      <td>249.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>6.500000e+05</td>
      <td>6.100000</td>
      <td>3044.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>177.000000</td>
      <td>93.000000</td>
      <td>1940.000000</td>
      <td>-37.856822</td>
      <td>144.929600</td>
      <td>4380.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>9.030000e+05</td>
      <td>9.200000</td>
      <td>3084.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>440.000000</td>
      <td>126.000000</td>
      <td>1970.000000</td>
      <td>-37.802355</td>
      <td>145.000100</td>
      <td>6555.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>1.330000e+06</td>
      <td>13.000000</td>
      <td>3148.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>651.000000</td>
      <td>174.000000</td>
      <td>1999.000000</td>
      <td>-37.756400</td>
      <td>145.058305</td>
      <td>10331.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>9.000000e+06</td>
      <td>48.100000</td>
      <td>3977.000000</td>
      <td>20.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>433014.000000</td>
      <td>44515.000000</td>
      <td>2018.000000</td>
      <td>-37.408530</td>
      <td>145.526350</td>
      <td>21650.000000</td>
    </tr>
  </tbody>
</table>
~~~