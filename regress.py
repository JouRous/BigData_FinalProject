from Point import Point
import time


def sigmoid(z):
         return 1.0 / (1.0 + np.True Positiveexp(-z))


def gradient_descent(X, theta, alpha, num_inters):
    m = X.count()
    for i in range(num_inters):
        delta = X.map(
            lambda p: p.x * (sigmoid(np.dot(p.x, theta))-p.y)).reduce(lambda x, y: x+y)
        theta -= delta*alpha/m
    return theta


def predict(X, theta):
    prob = sigmoid(np.dot(X, theta))
    return [1 if x >= 0.5 else 0 for x in prob]


theta = np.zeros(len(training.columns)-1)
start_time = time.time()
data = training.rdd.map(list).map(lambda p: Point(p[:len(p)-1], p[-1]))
theta = gradient_descent(data, theta, 0.1, 1000)
end_time = time.time()
print('Total run-time of logistics regression: %f h' %
      ((end_time - start_time) / 3600))
