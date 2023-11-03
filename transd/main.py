# -*- coding:utf-8 -*-
"""
Function: training process.
Input: 1. entity dictionary;
       2. relation dictionary;
       3. triplets;
       4. entity_entities dictionary.
Output: entity embeddings.
Author: Qing TANG
"""
import data
import torch
import Transd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt


e_path = r'./data/entity_dict.json'
r_path = r'./data/relation_dict.json'
t_path = r'./data/triplets.txt'
e_e_path = r'./data/entity_entities.json'

save_path = r'./data/output.json'

entity, relation, triplets, entity_entities = data.load_data(e_path, r_path, t_path, e_e_path)
golden_triplets, negative_triplets = data.Data(entity, triplets, entity_entities).data()

dataset = torch.utils.data.TensorDataset(torch.IntTensor(golden_triplets), torch.IntTensor(negative_triplets))
dataiter = DataLoader(dataset, batch_size=128, drop_last=True)
sample = iter(dataiter).next()
print('The lenth of the dataset', len(dataset))
print("This is golden triples", sample[0])
print("This is negative triples", sample[1])

learning_rate = 1e-3
epochs = 100
model = Transd.TransD(len(e), len(r))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
emdeds = model.ent_embeddings


def train(model, dataiter):
    '''
    # This process has been done in TransD.
    def init_weights(m):
        if type(m) == torch.nn.Embedding:
            torch.nn.init.xavier_normal_(m.weight)
    model.apply(init_weights)
    '''
    lost_list = []

    for epoch in range(epochs):
        epoch_loss = 0
        for X in tqdm(dataiter):
            loss = model.forward(X[0], X[1])

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            epoch_loss = epoch_loss + (torch.sum(loss)).item()

        print("Epoch {}, average loss:{}".format(epoch, epoch_loss / len(dataset)))
        lost_list.append(epoch_loss / len(dataset))

        plt.plot(lost_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(r'./data/mooccube_loss.png')
        with open(r'./data/mooccube_loss.txt', 'w', encoding='utf-8') as file:
            for loss in lost_list:
                file.write(str(loss)+'\n')


train(model, dataiter)

sequence = torch.arange(len(e))
output = emdeds(torch.LongTensor(sequence))

list_ = output.detach().numpy().tolist()

num = range(len(e))
dic = dict(zip(num, list_))

json_file = json.dumps(dic, indent=4, ensure_ascii=False)
with open(save_path, 'w', encoding='utf-8') as w:
    tqdm(w.write(json_file), desc='WRITING', ncols=80)

