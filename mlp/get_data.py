# -*- coding:utf-8 -*-
"""
Function: data creator.
Input: 1. user and course embeddings from TransD output;
       2. user and course embeddings from autoencoder output;
       3. user-course inetractions.
Output: training data and testing data.
Author: Qing TANG
"""
import json
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(transd_output, ae_output, user_courses):
    # Extract embeddings of users and resources from transd output
    with open(transd_output, 'r', encoding='utf-8') as trans:
        transd_dict = json.load(trans)

    # Extract embeddings of resources from ae output
    with open(ae_output, 'r', encoding='utf-8') as ae:
        ae_dic = json.load(ae)

    course_dict = {}
    # Splice the course vectors of ae and transd separately.
    for course in ae_dic.keys():
        course_dict[course] = ae_dic[course] + transd_dict[course]

    # Splice all course embeddings  706
    uni_course = []
    for vector in course_dict.values():
        uni_course.extend(vector)

    user_vectors_dict = {key: value for key, value in transd_dict.items() if int(key) < 2000}

    with open(user_courses, 'r', encoding='utf-8') as uc:
        dic = json.load(uc)

    users = list(dic.keys())  # All users who have enrolled course

    resources = list(dic.values())  # Courses enrolled by users, one-to-many


    X = np.array(users, dtype=object)
    y = np.array(resources, dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test, user_vectors_dict, uni_course


def tag(y_train, y_test):
    train_tags = []
    test_tags = []

    for list_ in y_train:
        a = np.zeros(shape=(1, 706), dtype=int)
        for pos in list_:
            if pos != 0:
                index = pos - 2000
                a[0][index] = 1
        train_tags.append(a)
    train_tags_array = np.array(train_tags).squeeze()

    for list_ in y_test:
        b = np.zeros(shape=(1, 706), dtype=int)
        for pos in list_:
            if pos != 0:
                index = pos - 2000
                b[0][index] = 1
        test_tags.append(b)
    test_tags_array = np.array(test_tags).squeeze()

    return train_tags_array, test_tags_array


def vector(X_train, X_test, user_vectors, uni_course_vec):
    train = []
    for i in range(len(X_train)):
        single = user_vectors[X_train[i]] + uni_course_vec
        train.append(single)
    train_array = np.array(train)
    print(train_array)
    print("The shape of the training data：{}".format(train_array.shape))

    test = []
    for i in range(len(X_test)):
        single = user_vectors[X_test[i]] + uni_course_vec
        test.append(single)
    test_array = np.array(test)
    print(test_array)
    print("The shape of the testing data：{}".format(test_array.shape))

    return train_array, test_array
