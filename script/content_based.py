from script import load_data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def Sort(sub_li):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key=lambda x: x[1])
    return list(reversed(sub_li))

df = load_data.load_data()
df.fillna(0, inplace=True)
df = shuffle(df).reset_index(drop=True)
df = df.truncate(before=0, after=10000)

Y = df['target'].tolist()
df1 = df.drop(['target'],axis=1)
X = df1.values.tolist()

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

def compute_similarity(item, x_train):
    similarities = []
    for i in range(len(x_train)):
        similarity = cosine_similarity([item], [x_train[i]])
        if similarity == 1:
            print("UGUALI?")
            print(item)
            print(x_train[i])
        similarities.append([i, similarity])
    return similarities

def predict(item, x_train, y_train, KNN):
    similarities = compute_similarity(item, x_train)
    sortedSimilarities = Sort(similarities)
    topK = sortedSimilarities[:KNN]
    zeros = 0
    ones = 0
    for elem in topK:
        index = elem[0]
        target = y_train[index]
        if target == 0:
            zeros += 1
        else:
            ones += 1
    total = zeros + ones
    zero_prob = zeros / total
    one_prob = 1 - zero_prob
    if one_prob >= zero_prob:
        return 1
    else:
        return 0

def predict2(item, x_train, y_train, KNN):
    similarities = compute_similarity(item, x_train)
    sortedSimilarities = Sort(similarities)
    topK = sortedSimilarities[:KNN]
    zeros = 0
    ones = 0
    for i in range(len(topK)):
        index = topK[i][0]
        target = y_train[index]
        if target == 0:
            zeros += (KNN - i)
        else:
            ones += (KNN - i)
    total = zeros + ones
    zero_prob = zeros / total
    one_prob = 1 - zero_prob
    if one_prob >= zero_prob:
        return 1
    else:
        return 0

def accuracy(x_train, x_test, y_train, y_test, KNN):
    score1 = 0
    score2 = 0
    total = len(y_test)
    for i in range(len(x_test)):
        print("PROCESSING ITEM No: " + str(i))
        if predict(x_test[i], x_train, y_train, KNN) == y_test[i]:
            score1 += 1
        if predict2(x_test[i], x_train, y_train, KNN) == y_test[i]:
            score2 += 1
    return score1/total, score2/total

print(accuracy(x_train, x_test, y_train, y_test, 10))


