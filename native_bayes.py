class NativeBayes:
    def __init__(self, train, test, label):
        self.train = train
        self.test = test
        self.label = label

    def mean(self, train_y, numbers):
        return (train_y.groupBy().mean(numbers).collect())[0][0]

    def stdev(self, train_y, numbers):
        avg = self.mean(train_y, numbers)
        lendf = len(train_y.select(numbers).collect())
        value = train_y.select(numbers).rdd.flatMap(lambda x: x).collect()
        variance = sum([(x-avg)**2 for x in value]) / float(lendf - 1)
        return sqrt(variance)

    def summarize_dataset(self, train_y):
        summaries = [(self.mean(train_y, column), self.stdev(train_y, column), train_y.select(column).count())
                     for column in train_y.columns]
        del(summaries[-1])
        return summaries

    def summarize_by_class(self, train):
        summaries = dict()
        lenqual = len(train.select([self.label]).rdd.flatMap(
            lambda x: x).distinct().collect())
        for i in range(0, lenqual):
            item = train.select([self.label]).rdd.flatMap(
                lambda x: x).distinct().collect()[i]
            train_y = train.filter(train[self.label] == item)
            summaries[item] = self.summarize_dataset(train_y)
        return summaries

    def calculate_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / \
                float(total_rows)
            for i in range(0, len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(
                    row[i], mean, stdev)
        return probabilities

    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def predict(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def fit(self):
        summarize = self.summarize_by_class(self.train)
        predictions = list()
        for row in self.test.collect():
            output = self.predict(summarize, row)
            predictions.append(output)
        return(predictions)
