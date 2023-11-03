# -*- coding:utf-8 -*-
"""
Function: 1. define classes of entity, relation, and attribute;
          2. extract entities from interactions;
          3. make triplets - <h,r,t>.
Input: interactions
Output: 1. entity dictionary;
        2. relation dictionary;
        3. triplets.
Author: Qing TANG
"""
import json
import tqdm
from collections import Counter


class Extractor:
    def __init__(self):
        super(Extractor, self).__init__()
        self.user = set()
        self.course = set()
        self.teacher = set()
        self.school = set()
        self.video = set()
        self.kc = set()

        self.triplets = []

        self.user_counter = []
        self.kc_counter = []
        self.video_counter = []

    def extract_entity(self, path_list, entity_dict_path):
        print('-------------user-course.json----------------')
        with open(path_list[0], 'r', encoding='utf-8') as u_c:
            for line in tqdm.tqdm(u_c.readlines(), desc='user-course', ncols=80):
                line = line.split()
                self.user.add(line[0].strip())
                self.course.add(line[1].strip())

                self.user_counter.append(line[0].strip())  # used for verifying the number of interactions

        print('-------------teacher-course.json---------------------')
        with open(path_list[1], 'r', encoding='utf-8') as t_c:
            for line in tqdm.tqdm(t_c.readlines(), desc='teacher-course', ncols=80):
                line = line.split()
                self.teacher.add(line[0].strip())
                self.course.add(line[1].strip())

        print('-------------school-course.json---------------------')
        with open(path_list[2], 'r', encoding='utf-8') as s_c:
            for line in tqdm.tqdm(s_c.readlines(), desc='school-course', ncols=80):
                line = line.split()
                self.school.add(line[0].strip())
                self.course.add(line[1].strip())

        print('-------------school-teacher.json----------------------')
        with open(path_list[3], 'r', encoding='utf-8') as s_t:
            for line in tqdm.tqdm(s_t.readlines(), desc='school-teacher', ncols=80):
                line = line.split()
                self.school.add(line[0].strip())
                self.teacher.add(line[1].strip())

        print('-------------course-concept.json--------------------')
        with open(path_list[4], 'r', encoding='utf-8') as c_c:
            for line in tqdm.tqdm(c_c.readlines(), desc='course-concept', ncols=80):
                line = line.split()
                self.course.add(line[0].strip())
                self.kc.add(line[1].strip())

                self.kc_counter.append(line[1].strip())

        print('-------------user-video.json ----------------')
        with open(path_list[5], 'r', encoding='utf-8') as u_v:
            for line in tqdm.tqdm(u_v.readlines(), desc='user-video', ncols=80):
                line = line.split()
                self.user.add(line[0].strip())
                self.video.add(line[1].strip())

                self.video_counter.append(line[1].strip())

        print('-------------course-video.json--------------------')
        with open(path_list[6], 'r', encoding='utf-8') as c_v:
            for line in tqdm.tqdm(c_v.readlines(), desc='course-video', ncols=80):
                line = line.split()
                self.course.add(line[0].strip())
                self.video.add(line[1].strip())

        print('-------------video-kc.json--------------------')
        with open(path_list[7], 'r', encoding='utf-8') as v_k:
            for line in tqdm.tqdm(v_k.readlines(), desc='video-kc', ncols=80):
                line = line.split()
                self.video.add(line[0].strip())
                self.kc.add(line[1].strip())

                self.kc_counter.append(line[1].strip())

        user_statistic = Counter(self.user_counter)
        user_data = sorted(user_statistic.items(), key=lambda x: x[1], reverse=True)[:2000]
        user = [x[0] for x in user_data]
        video_statistic = Counter(self.video_counter)
        video_data = sorted(video_statistic.items(), key=lambda y: y[1], reverse=True)[:5000]
        video = [y[0] for y in video_data]
        kc_statistic = Counter(self.kc_counter)
        kc_data = sorted(kc_statistic.items(), key=lambda z: z[1], reverse=True)[:5000]
        kc = [z[0] for z in kc_data]
        print('user number: {}'.format(len(user)))
        print('video number: {}'.format(len(video)))
        print('kc number: {}'.format(len(kc)))
        print('course number: {}'.format(len(self.course)))
        print('school number: {}'.format(len(self.school)))
        print('teacher number: {}'.format(len(self.teacher)))

        print('-------------create entity dictionary-------------------')
        keys = user + list(self.course) + list(self.school) + list(self.teacher) + video + kc
        entity_dict = dict(zip(keys, range(len(keys))))

        json_file = json.dumps(entity_dict, indent=4, ensure_ascii=False)
        with open(entity_dict_path, 'w', encoding='utf-8') as w:
            tqdm.tqdm(w.write(json_file))
        print('-------------finish entity dictionary-----------------')

        return entity_dict

    def extract_relation(self, relation_dict_path):
        print('-------------create relation dictionary-------------------')
        relations = ['enroll', 'refer', 'offer', 'teach', 'in', 'contain', 'watch']
        keys = relations
        relation_dict = dict(zip(keys, range(len(keys))))

        json_file = json.dumps(relation_dict, indent=4, ensure_ascii=False)
        with open(relation_dict_path, 'w', encoding='utf-8') as w:
            tqdm.tqdm(w.write(json_file))
        print('-------------finish relation dictionary-----------------')

        return relation_dict

    def create_triplets(self, path_list, entity_dict, relation_dict, save_path):
        print('-------------user-course.json----------------')
        zero = []
        with open(path_list[0], 'r', encoding='utf-8') as u_c:
            for line in tqdm.tqdm(u_c.readlines(), desc='user-course', ncols=80):
                line = line.split()
                if line[0] in entity_dict.keys():
                    zero.append(entity_dict[line[0]])
                    zero.append(relation_dict['enroll'])
                    zero.append(entity_dict[line[1]])
                    self.triplets.append(zero)
                    zero = []

        print('-------------teacher-course.json---------------------')
        one = []
        with open(path_list[1], 'r', encoding='utf-8') as t_c:
            for line in tqdm.tqdm(t_c.readlines(), desc='teacher-course', ncols=80):
                line = line.split()
                one.append(entity_dict[line[0]])
                one.append(relation_dict['teach'])
                one.append(entity_dict[line[1]])
                self.triplets.append(one)
                one = []

        print('-------------school-course.json---------------------')
        two = []
        with open(path_list[2], 'r', encoding='utf-8') as s_c:
            for line in tqdm.tqdm(s_c.readlines(), desc='school-course', ncols=80):
                line = line.split()
                two.append(entity_dict[line[0]])
                two.append(relation_dict['offer'])
                two.append(entity_dict[line[1]])
                self.triplets.append(two)
                two = []

        print('-------------school-teacher.json----------------------')
        three = []
        with open(path_list[3], 'r', encoding='utf-8') as s_t:
            for line in tqdm.tqdm(s_t.readlines(), desc='school-teacher', ncols=80):
                line = line.split()
                three.append(entity_dict[line[1]])
                three.append(relation_dict['in'])
                three.append(entity_dict[line[0]])
                self.triplets.append(three)
                three = []

        print('-------------course-concept.json--------------------')
        four = []
        with open(path_list[4], 'r', encoding='utf-8') as c_c:
            for line in tqdm.tqdm(c_c.readlines(), desc='course-concept', ncols=80):
                line = line.split()
                if line[1] in entity_dict.keys():
                    four.append(entity_dict[line[0]])
                    four.append(relation_dict['refer'])
                    four.append(entity_dict[line[1]])
                    self.triplets.append(four)
                    four = []

        print('-------------user-video.json ----------------')
        five = []
        with open(path_list[5], 'r', encoding='utf-8') as u_v:
            for line in tqdm.tqdm(u_v.readlines(), desc='user-video', ncols=80):
                line = line.split()
                if line[0] in entity_dict.keys() and line[1] in entity_dict.keys():
                    five.append(entity_dict[line[0]])
                    five.append(relation_dict['watch'])
                    five.append(entity_dict[line[1]])
                    self.triplets.append(five)
                    five = []

        print('-------------course-video.json--------------------')
        six = []
        with open(path_list[6], 'r', encoding='utf-8') as c_v:
            for line in tqdm.tqdm(c_v.readlines(), desc='course-video', ncols=80):
                line = line.split()
                if line[1] in entity_dict.keys():
                    six.append(entity_dict[line[0]])
                    six.append(relation_dict['contain'])
                    six.append(entity_dict[line[1]])
                    self.triplets.append(six)
                    six = []

        print('-------------video-kc.json--------------------')
        seven = []
        with open(path_list[7], 'r', encoding='utf-8') as v_k:
            for line in tqdm.tqdm(v_k.readlines(), desc='video-kc', ncols=80):
                line = line.split()
                if line[0] in entity_dict.keys() and line[1] in entity_dict.keys():
                    seven.append(entity_dict[line[0]])
                    seven.append(relation_dict['refer'])
                    seven.append(entity_dict[line[1]])
                    self.triplets.append(seven)
                    seven = []

        t = ''
        with open(save_path, 'w', encoding='utf-8') as w2:
            for items in self.triplets:
                for i in range(len(self.triplets[0])):
                    t = t + str(items[i]) + ' '
                w2.write(t.strip(' '))
                w2.write('\n')
                t = ''


if __name__ == '__main__':
    u_c_p = r'./data/relations/user-course.json'
    t_c_p = r'./data/relations/teacher-course.json'
    s_c_p = r'./data/relations/school-course.json'
    s_t_p = r'./data/relations/school-teacher.json'
    c_k_p = r'./data/relations/course-concept.json'
    u_v_p = r'./data/relations/user-video.json'
    c_v_p = r'./data/relations/course-video.json'
    v_k_p = r'./data/relations/video-concept.json'

    e_d_p = r'./data/entity_dict.json'
    r_d_p = r'./data/relation_dict.json'

    s_p = r'./data/triplets.txt'

    p_list = [u_c_p, t_c_p, s_c_p, s_t_p, c_k_p, u_v_p, c_v_p, v_k_p]

    e_d = Extractor().extract_entity(p_list, e_d_p)
    r_d = Extractor().extract_relation(r_d_p)
    triplets = Extractor().create_triplets(p_list, e_d, r_d, s_p)
