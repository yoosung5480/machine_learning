import numpy as np

class AdalineGD(object):

    """
    선형 뉴런 분석기

    매개변수
        eta: float
            0.0 ~ 0.1 사이의 학습률
        n_iter: int 
            훈련 데이터셋 반복횟수
        random_stata : int
            가중치 무작위 초기화를 위한 난수 생성 시드번호

    속성
        w_ : 1차원 배열
            학습된 가중치
        cost : list
            데이터셋 마다의 비용 제곱합
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
    # param 
    # X : shape(n_samples, n_features)
    #   n개의 샘플과 n개의 특성으로 이뤄진 훈련 데이터
    # y : shape(1, n_samples) 
    #   타깃

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # numpy의 randomstate 객체를 반환한다. 이를 이용하면 일관된 난수 시퀸스를 생성하게 된다.
        # 첫번째 원소는 편향값이다. 이 가중치들을 0으로 초기화 하게되면 가중치 벡터의 방향에는 영향을 주지못하고, 크기에만 영향을 미친다.
        self.cost_ = []

        for i in range(self, X):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    # y = x 꼴의 간단한 선형 활성화 함수
    def activation(self, X):
        return X

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] # 배열 X, W[1:]의 내적계산 후 편향값 더하기
    

    def predict(self, X):
        # 단위 계단함수를 사용하여 클래쓰 레이블은 만든다.
        return np.where(self.net_input(X) >= 0.0, 1, -1) # 위의 인풋값에 대한 출력값을 매개로 해서, 만약 값이 0보다 크다면 1, 아니면 -1 출력