# -*- coding:utf-8 -*-
"""
Function: count one-to-many from triples.
Input: triplets.
Output: 1. entity-entities;
        2. user-courses.
Author: Qing TANG
"""
from collections import defaultdict
from tqdm import tqdm
import json


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def create_one2many(triplets_path, entity_items_path, user_courses_path):
    entity_entities = defaultdict(set)
    user_courses = defaultdict(set)

    with open(triplets_path, 'r') as r:
        for line in tqdm(r.readlines()):
            u, _, i = line.split(" ")
            u = int(u)
            i = int(i)
            entity_items[u].add(i)
            if (u < 2000) and (2000 <= i <= 2705):
                user_courses[u].add(i)
             
json_file_1 = json.dumps(entity_entities, indent=4, ensure_ascii=False, cls=SetEncoder)
with open(entity_items_path, 'w', encoding='utf-8') as w:
    tqdm(w.write(json_file_1), desc='WRITING', ncols=80)

json_file_2 = json.dumps(user_courses, indent=4, ensure_ascii=False, cls=SetEncoder)
with open(user_courses_path, 'w', encoding='utf-8') as w:
    tqdm(w.write(json_file_2), desc='WRITING', ncols=80)


if __name__ == '__main__':
    t_p = r'./data/triplets.txt'
    e_i_p = r'.\data\entity_entities.json'
    u_c_p = r'.\data\user_courses.json'
    create_one2many(t_p, e_i_p, u_c_p)