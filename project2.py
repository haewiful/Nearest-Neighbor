import pandas as pd
import sys
import math
from scipy.stats import zscore
import scipy.stats

def main():
    # print initial info. and receive filename and 
    print("Welcome to Bertie Woosters Feature Selection Algorithm")
    filename = input("Type in the name of the file to test: ")

    print("\nType the number of the algorithm you want to run.\n")
    print("\t1) Forward Selection")
    print("\t2) Backward Elimination\n")
    selection = int(input())

    # read data
    data = pd.read_csv(filename, header=None, sep='  ', engine='python')

    # do z-normalization
    for col in data.columns:
        if col==0:
            continue
        data[col] = (data[col]-data[col].mean())/data[col].std(ddof=0)

    # overall info
    print(f"This dataset has {len(data.loc[0])-1} features (not including the class attribute), with {len(data)} instances.\n")

    # Do a nearest neighbor with all features
    print(f"""Running nearest neighbor with all {len(data.loc[0])-1} features, using "leaving-one-out" evaluation, I get an accuracy of {nearest_neighbor(data, [x for x in range(1, len(data.loc[0]))]):.1f}%""")

    # according to the choice, do leaving-one-out evaluation
    if selection == 1:
        features = [[x] for x in range(1, len(data.loc[0]))]
    else:
        features = [[x for x in range(1, len(data.loc[0]))]]
    
    print("\nBeginning search.\n")

    max_accuracy = -1
    max_set = []
    local_max_acc = -1
    local_max_set = []
    index = 0
    while(True):
        feat = features[index]
        index+=1
        acc = nearest_neighbor(data, feat)
        print(f"\t\tUsing feature(s) {set_to_string(feat)} accuracy is {acc:.1f}%")
        if acc > local_max_acc:
            local_max_acc = acc
            local_max_set = feat
        
        # no more feature
        if len(features) == index:
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
                if len(features[0]) == len(data.loc[0])-1:
                    break
                features = []
                for i in range(1, len(data.loc[0])):
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
    
    print(f"Finished search!! The best feature subset is {set_to_string(max_set)}, which has an accuracy of {max_accuracy:.1f}%")

def nearest_neighbor(data, feature):
    correct_count=0
    for test_index, test_row in data.iterrows():
        # print(f"\n=== testing row {test_index} ===")
        min_distance=sys.maxsize
        min_index=-1
        for index, row in data.iterrows():
            if test_index == index:
                continue
            distance = calc_distance(data, test_index, index, feature)
            # print(f"{index} - {distance}")

            # update nearest neighbor
            if distance < min_distance:
                min_distance = distance
                min_index = index
        
        # print(f"nearest neighbor is {min_index} - {min_distance}")
        
        # check correctness
        if test_row[0] == data.loc[min_index][0]:
            # print(f"same class of {test_row[0]} == {data.loc[min_index][0]}")
            correct_count+=1
    # print(f"correct {correct_count} out of {len(data)}")
    accuracy = correct_count / len(data)*100

    return accuracy

def calc_distance(data, t_index, index, feature_set):
    test = data.loc[t_index]
    row = data.loc[index]

    test_feat = []
    row_feat = []
    for i in feature_set:
        test_feat.append(test[i])
        row_feat.append(row[i])
    distance = math.dist(test_feat, row_feat)
    return distance

def set_to_string(f_set):
    tmp = "{"
    for x in f_set:
        tmp += str(x) + ","
    return tmp[:-1] + "}"



if __name__=="__main__":
    main()