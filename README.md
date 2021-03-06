# CV
컴퓨터 비전(computer vision)


## 프로젝트 명 : 빨간 공 찾기
## 프로젝트 기간 : 2018.11 ~ 2018.12
## 개발 단계 : 완성
## 기술 스택 : C++, openCV
## 소스 코드 : github.com/ssjf409/CV/tree/master/app/assignment/term_project


## 프로젝트 내용
### 개발 의도

디즈니 애니메이션에서는 실제로 연기를 하는 배우와 컴퓨터 그래픽의 이미지를 합치는 과정을 통해서 애니메이션을 만든다고 한다. 이 때, 다양한 각도의 영상을 얻기 위해서 여러 대의 카메라를 사용하는데, 다중 카메라의 좌표계를 일치시키기 위해서는 다중 카메라로 획득한 영산간의 대응쌍(match or correspondence)을 찾아야 한다. 이를 위해서 단일 컬러로 채색된 sphere object가 많이 사용된다. 서로 다른 두 영상에서 동일한 sphere object를 각각 정확하게 검출(detection) 할 수 있으면, 대응점 탐색(correspondence search) 및 정합 (matching) 과정 없이 검출만으로 두 영상 간의 대응쌍을 찾을 수 있게 된다. 이 때, 실시간으로 적용이 가능하도록 circle Hough transform보다 속도와 정확도 측면에서 성능이 우수한 알고리즘을 만들어 보는 것이다.

 

### 알고리즘

![image](https://user-images.githubusercontent.com/35087350/83962384-d39df180-a8d7-11ea-98fe-c76164548b5c.png)
![image](https://user-images.githubusercontent.com/35087350/83962391-dd275980-a8d7-11ea-95c3-d80209d1562f.png)




1. 이미지에서 특정색(sphere object의 색)을 제외하고 전부 마스크 처리 하여, 전부 값을 지운다. (그림2)

 

2. 공의 빈 공간을 채우고 공 밖에 있는 노이즈(점)들을 조금이라도 지우기 위해 GaussianBlur처리를 해준다. (그림3)

 

3. 그림3에 있는 모든 픽셀들의 y와 x에 대해 각각 위치값을 누적합으로 구한다. 그 후 더해진 갯수 만큼 나누면 중앙 값이 구해진다. (그림4)

 

소스 코드 : https://github.com/ssjf409/CV/blob/master/app/assignment/term_project/circle_detection.cpp

int sumY = sumX = 0;
int cnt = 0;
for(int i = 0; i < height; i++) {
  for(int j = 0; j < width; j++) {
    if( [i, j] 위치에 값이 존재한다면) {
      sumY += i;
      sumX += j;
      cnt++;
    }
  }
}
double centerY = (double)sumY / cnt;
double centerX = (double)sumX / cnt;
 

4. 이후 반지름은 원의 중심점을 기준으로 상, 하, 좌, 우 4방향으로 거리를 1씩 늘리면서 마지막으로 값이 조재하는 지점을 저장한다. 연산이 다 끝나면 각 방향별로 구해진 거리를 평균으로 반지름을 구한다. (그림5)

 

 

### 알고리즘의 한계

 

1. 색상을 기반으로 원을 검출하므로, 조명의 영향을 많이 받는다.

 

2. 반지름을 구할 때 상, 하, 좌, 우 4방향으로 계산을 한 뒤 평균을 하기 때문에 오차가 있을 수 있다. 이는 sin, cos으로 16방향으로 늘린 후 최소값 및 최대값을 제외하고 평균을 구하면 오차가 더욱 줄어 들 수 있을 것이다.

 

### 시도했던 다른 알고리즘

 


아이디어 : 원 내부에 직각 삼각형을 그리면 직각 삼각형의 무게 중심은 항상 원의 중심점이 된다.

![image](https://user-images.githubusercontent.com/35087350/83962399-e6b0c180-a8d7-11ea-96f5-930daedac54d.png)


1. line dectection으로 경계면들을 구해준다.

2. 값이 존재 하는 모든 픽셀들 마다, 동일한 y값을 갖지만 x값이 다른 좌표와 동일한 x값을 갖지만 y값이 다른 좌표를 구해서 만들 수 있는 모든 삼각형을 만든다.

3. 그리고 이 삼각형들의 무게 중심을 구한 뒤 그 값들을 이차원 배열로 카운트한다.

 

제대로 동작하지 않은 이유

원과 상관없이 경계면의 밀집도가 높은 부분은 카운트가 높게 나오는 현상을 보였다.


좋아요공감
공유하기통계글 요소
저작자표시
