import pandas as pd
import sys
import math
from scipy.stats import zscore
import scipy.stats
import time
import numpy as np
import threading
import queue

def main():
    # print initial info. and receive filename and 
    print("Welcome to Bertie Woosters Feature Selection Algorithm")
    filename = input("Type in the name of the file to test: ")

    print("\nType the number of the algorithm you want to run.\n")
    print("\t1) Forward Selection")
    print("\t2) Backward Elimination\n")
    selection = int(input())

    # read data & normalize it
    data = get_data(filename)
    # print(data[0]) # row
    # print(data[:, 2]) # column

    # overall info
    print(f"This dataset has {len(data[0])-1} features (not including the class attribute), with {len(data)} instances.\n")

    # Do a nearest neighbor with all features
    start_time = time.time()
    que = queue.Queue()
    full_acc, _ = nearest_neighbor(que, data, [x for x in range(1, len(data[0]))])
    print(f"""Running nearest neighbor with all {len(data[0])-1} features, using "leaving-one-out" evaluation, I get an accuracy of {full_acc:.1f}%""")

    # according to the choice, do leaving-one-out evaluation
    if selection == 1:
        features = [[x] for x in range(1, len(data[0]))]
    else:
        features = [[x for x in range(1, len(data[0]))]]
    
    print("\nBeginning search.\n")

    max_accuracy = -1
    max_set = []
    local_max_acc = -1
    local_max_set = []
    index = 0
    que.queue.clear()
    while(True):
        # start thread
        threads = []
        # print("Features", features)
        for feat in features:
            if len(feat) == len(data[0])-1: # already done this
                local_max_acc = full_acc
                local_max_set = [x for x in range(1, len(data[0]))]
                print(f"\t\tUsing feature(s) {set_to_string(local_max_set)} accuracy is {local_max_acc:.1f}%")
            else:
                tmp_thread = threading.Thread(target=nearest_neighbor, args=(que, data, feat))
                tmp_thread.start()
                threads.append(tmp_thread)
                # acc = nearest_neighbor(data, feat)
            # print(f"\t\tUsing feature(s) {set_to_string(feat)} accuracy is {acc:.1f}%")
        for th in threads:
            th.join()
        threads.clear()
        # end thread (join)

        # update local_max -> critical section
        while not que.empty():
            acc, feat = que.get()
            if acc > local_max_acc:
                local_max_acc = acc
                local_max_set = feat
        
        # no more feature
        print()
        # update max_accuracy
        if local_max_acc > max_accuracy:
            max_accuracy = local_max_acc
            max_set = local_max_set
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        print(f"Feature set {set_to_string(local_max_set)} was best, accuracy is {local_max_acc:.1f}%\n")

        # check if we checked all feature sets
        # update features
        if selection==1:
            if len(features[0]) == len(data[0])-1:
                break
            features = []
            for i in range(1, len(data[0])):
                if i not in local_max_set:
                    features.append(local_max_set+[i])
        else:
            if len(features[0]) == 1:
                break
            features = []
            for i in range(0, len(local_max_set)):
                tmp = local_max_set.copy()
                tmp.pop(i)
                features.append(tmp)
            
        # reset variables
        local_max_acc = -1
        local_max_set = []
        index=0
    end_time = time.time()
    print(f"Finished search!! The best feature subset is {set_to_string(max_set)}, which has an accuracy of {max_accuracy:.1f}%")
    print(f"This took: {end_time-start_time} seconds")

def nearest_neighbor(queue, data, feature):
    correct_count=0
    distance_set = []

    for test_index, test_row in enumerate(data):
        distance_tmp = []
        # print(f"=== testing row {test_index} ===")
        min_distance=sys.maxsize
        min_index=-1
        for index, row in enumerate(data):
            if test_index == index:
                distance_tmp.append(0)
                continue
            if test_index < index:
                distance = calc_distance(data, test_index, index, feature)
            else:
                distance = distance_set[index][test_index]
            distance_tmp.append(distance)
            # print(f"{index} - {distance}")

            # update nearest neighbor
            if distance < min_distance:
                min_distance = distance
                min_index = index
        # print(distance_tmp)
        distance_set.append(distance_tmp)
        
        # print(f"nearest neighbor is {min_index} - {min_distance}")
        
        # check correctness
        if test_row[0] == data[min_index][0]:
            # print(f"same class of {test_row[0]} == {data.loc[min_index][0]}")
            correct_count+=1
    # print(f"correct {correct_count} out of {len(data)}")
    accuracy = correct_count / len(data)*100
    print(f"\t\tUsing feature(s) {set_to_string(feature)} accuracy is {accuracy:.1f}%")
    queue.put((accuracy, feature))
    return accuracy, feature

def calc_distance(data, t_index, index, feature_set):
    test = data[t_index][feature_set]
    row = data[index][feature_set]
    distance = math.dist(test, row)
    return distance

def get_data(filename):
    data = pd.read_csv(filename, header=None, sep='  ', engine='python')

    # do z-normalization
    for col in data.columns:
        if col==0:
            continue
        data[col] = (data[col]-data[col].mean())/data[col].std(ddof=0)
    return data.to_numpy()

def set_to_string(f_set):
    tmp = "{"
    for x in f_set:
        tmp += str(x) + ","
    return tmp[:-1] + "}"

if __name__=="__main__":
    main()
