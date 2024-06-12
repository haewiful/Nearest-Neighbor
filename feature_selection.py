import pandas as pd
import sys
import math


def calc_distance(t_index, index, feature_set):
    test = data.loc[t_index]
    row = data.loc[index]

    test_feat = []
    row_feat = []
    for i in feature_set:
        test_feat.append(test[i])
        row_feat.append(row[i])
    # for ind in range(1, f_num+1):
    #     print(test.loc[ind], end=', ')
    # print()
    # for ind in range(1, f_num+1):
    #     print(row.loc[ind], end=', ')
    # print()
    distance = math.dist(test_feat, row_feat)
    print(f"{index} : {distance}")
    return distance
    # distance = math.dist(test)



# read data file
# data = pd.read_csv('CS205_small_Data__27.txt', sep='  ', engine='python')
# data = pd.read_csv('test1.txt', header=None, sep='  ', engine='python')
data = pd.read_csv('test2.txt', header=None, sep='  ', engine='python')
# print(data.to_string())

# print(len(data)) # row num
# print(len(data.loc[0])) # col num

# print(type(data.loc[0]))

# run on all features

# start searching

features = [[x] for x in range(1, len(data.loc[0]))]
previous=[[x] for x in range(1, len(data.loc[0]))]
# print(previous)
for _ in range(2, len(data.loc[0])):
    prev_tmp = []
    for cur_set in previous:
        # print(cur_set)
        for i in range(1, len(data.loc[0])):
            if i not in cur_set:
                tmp = cur_set + [i]
                tmp.sort()
                if tmp not in features:
                    prev_tmp.append(tmp)
                    features.append(tmp)
    previous = prev_tmp

# print("features\n", features)
# exit()

for cur_feature in features:
correct_count=0

for test_index, test_row in data.iterrows():
    print(f"=== testing row {test_index} ===")
    min_distance=sys.maxsize
    min_index=-1
    for index, row in data.iterrows():
        if test_index==index:
            continue
        distance = calc_distance(test_index, index, 3)

        # update nearest neighbor
        if distance < min_distance:
            min_distance = distance
            min_index = index
    # check accuracy
    print(f"nearest - {min_index} : {min_distance}")
    if test_row[0] == data.loc[min_index][0]:
        print(f"same class of {test_row[0]} == {data.loc[min_index][0]}")
        correct_count+=1
    print()

print(f"correct {correct_count} out of {len(data)}")
print(f"accuracy = {correct_count/len(data)*100}")

# leaving-one-out evaluation (for all data)
    # find the nearest neighbor for that data
    # check if classification is correct & update result