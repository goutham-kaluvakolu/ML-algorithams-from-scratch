import numpy as np
import pandas as pd

a = pd.read_csv("hw2_3c.csv")
a = a.iloc[:, [0, 1, 2, 3]].to_numpy()
a = np.array(a, dtype="float64")
a = np.sort(a, axis=0)


def get_thresholds(rows, columns):
    all_th = []
    thresholds = []
    for i in range(columns):
        for j in range(rows-1):
            avg = a[j][i]+a[j+1][i]
            thresholds.append(avg/2)
        all_th.append(thresholds)
        thresholds = []
    count = 0
    for i in all_th:
        for j in i:
            # print(j)
            continue
        # print("********")
    return all_th


def entropy(count):
    num = np.sum(count)
    val = 0
    for i in count:
        p = i/num
        val += -(p*np.log2(p))
    return val


def ifgain(count1, count2):
    root = entropy([2, 2, 2])
    w = weightedavg(count1, count2)
    return root-w


def weightedavg(count1, count2):
    split = [sum(count1), sum(count2)]
    c1 = entropy(count1)
    c2 = entropy(count2)
    w = ((split[0]*c1)+(split[1]*c2))/sum(split)
    return w


def lismaker(lis):
    s1 = lis
    dic_s1 = {}
    for i in s1:
        if i in dic_s1:
            dic_s1[i] += 1
        else:
            dic_s1[i] = 1
    num_lis_s1 = []
    for i, j in dic_s1.items():
        num_lis_s1.append(j)
    return num_lis_s1


def split(lis1, lis2):
    s1 = lismaker(lis1)
    s2 = lismaker(lis2)
    return ifgain(s1, s2)


def split_2_classes(col, th, label):
    class1 = col.loc[col[label] <= th, ['material']].to_numpy().flatten()
    class2 = col.loc[col[label] > th, ['material']].to_numpy().flatten()
    return split(class1, class2)


def node(rows, columns, data, root):
    all_th = get_thresholds(rows, columns)
    all_gains = []
    curr_max = 0
    for index, i in enumerate(all_th):
        label = data.columns[index]
        feature = data[[label, "material"]]
        feature = feature.sort_values(by=[label])
        gains = []
        for j in i:
            gains.append(split_2_classes(feature, j, label))
        if np.max(gains) > curr_max:
            curr_max = np.max(gains)
        all_gains.append(gains)
    curr_th = np.argwhere(all_gains == curr_max)[0]
    f = curr_th[0]
    rang = curr_th[1]
    label = data.columns[f]
    th = all_th[f][rang]
    root.label = label
    root.th = th
    data_split_left = data.loc[data[label] <= th]
    data_split_right = data.loc[data[label] > th]
    return data_split_left, data_split_right


columns = 3
data = pd.read_csv("hw2_3c.csv")
rows = data.shape[0]
curr_lvl = 0


class TreeNode:
    def __init__(self, val, left=None, right=None, pred=None):
        self.val = val
        self.left = left
        self.right = right
        self.prediction = pred
        self.th = 0
        self.label = ''


tree = []


def get_pred(data):
    x = data["material"].mode().to_numpy()
    return x


def dt(lvl, root, curr_lvl):
    if not root:
        print("compinling tree.....working on level 1")
        root = TreeNode(data, None, None, None)
        pred = get_pred(data)
        x, y = node(rows, columns, data, root)
        xx = TreeNode(x, None, None, get_pred(x))
        yy = TreeNode(y, None, None, get_pred(y))
        root.left = xx
        root.right = yy
        root.prediction = pred
        tree.append(root)
        curr_lvl += 2
        dt(lvl, root.left, curr_lvl)
        dt(lvl, root.right, curr_lvl)

    if root.val.empty:
        return

    if curr_lvl >= lvl:
        return
    print("compinling tree.....working on ", curr_lvl-1, "level")
    x, y = node(rows, columns, root.val, root)
    xx = TreeNode(x, None, None, get_pred(x))
    yy = TreeNode(y, None, None, get_pred(y))
    root.left = xx
    root.right = yy
    curr_lvl += 1
    dt(lvl, root.left, curr_lvl)
    dt(lvl, root.right, curr_lvl)



dic = {'height': 0, 'diameter': 1, 'weight': 2}
result = []
lvl = 8


def prediction(data, root, i_lvl, lvl):
    if not root:
        return
    if lvl == i_lvl:
        return root.prediction
    if lvl > i_lvl:
        d_col = root.label
        d_th = root.th
        d_index = dic[d_col]
        val = data[d_index]
        i_lvl += 1
        if val <= d_th:
            return prediction(data, root.left, i_lvl, lvl)
        else:
            return prediction(data, root.right, i_lvl, lvl)


test_data = pd.read_csv("hw2_3c_test.csv")
target = test_data['material'].to_numpy()


train_acc = []
test_acc = []


def get_acc(data, target, t_data, t_target):
    for i in range(1, 9, 1):
        f_result = []
        f1_result = []
        dt(i+1, None, 0)
        root = tree[0]
        print("creating the decision tree of depth: ", i)

        for row in data[['height', 'diameter', 'weight']].to_numpy():
            result = prediction(row, root, 1, i)
            f_result.append(result[0])
            # print(result[-1][0])
            result = []

        for row in t_data[['height', 'diameter', 'weight']].to_numpy():
            result = prediction(row, root, 1, i)
            f1_result.append(result[0])
            # print(result[-1][0])
            result = []

        count = 0
        for i, j in zip(f_result, target):
            # print("i,j",i,j)
            if i == j:
                count += 1
        train_acc.append(count/data.shape[0])

        count = 0
        for i, j in zip(f1_result, t_target):
            # print("i,j",i,j)
            if i == j:
                count += 1
        test_acc.append(count/t_data.shape[0])

        tree.pop()


def get_targets(data):
    return data['material'].to_numpy()


target = get_targets(data)
test_target = get_targets(test_data)
get_acc(data, target, test_data, test_target)
print("train accuracy:", train_acc)
print("test accuracy:", test_acc)
