# -*- coding:utf-8 -*-
"""
Function: taining and evaluating processes.
Input: 1. user and course embeddings from TransD output;
       2. user and course embeddings from autoencoder output;
       3. user-course inetractions.
Output: trained model and evaluation results.
Author: Qing TANG
"""
import torch
import numpy
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from MLP import MLP
import get_data


def dataiter(): # transd_path, ae_path, user_courses_path
    transd_path = r'./data/transd.json'
    ae_path = r'./data/ae.json'
    user_courses_path = r'./data/user_courses.json'

    X_train, X_test, y_train, y_test, user_vectors, uni_course_vec = get_data.load_data(transd_path, ae_path, user_courses_path)
    train_tags, test_tags = get_data.tag(y_train, y_test)

    train_array, test_array = get_data.vector(X_train, X_test, user_vectors, uni_course_vec)

    train_tensor = torch.from_numpy(train_array)
    # Note that you cannot directly use torch.Tensor() to convert array to tensor. You need to use torch.from_numpy(), otherwise all values will become 0.
    test_tensor = torch.from_numpy(test_array)

    train_tags_tensor = torch.from_numpy(train_tags).float()  # Note that the data type is changed here to float
    test_tags_tensor = torch.from_numpy(test_tags).float()

    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_tags_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_tags_tensor)

    train_dataiter = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_dataiter = DataLoader(test_dataset, batch_size=64, shuffle=True)
    sample = (iter(train_dataiter).next())
    print(sample)

    return train_dataiter, test_dataiter, train_dataset


def train(train_data, test_data, train_dataset):
    model = MLP()
    model = model.float()
    '''
    use_cuda = torch.cuda.is_available()
    print('use_cuda', use_cuda)

    if use_cuda is True:
       train_data = train_data.cuda()
       test_data = test_data.cuda()
       model = model.cuda()
    '''
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)

    model.apply(init_normal)

    learning_rate = 1e-3
    epoch = 100

    criteon = nn.BCELoss()
    # criteon = nn.MultiLabelSoftMarginLoss()
    # criteon = F.binary_cross_entropy_with_logits()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print(model)

    total_train_step = 0
    single_lost_list = []
    epoch_loss_list = []
    hr_list = []
    ndcg_list = []
    hr_ndcg_dict = {}

    for i in range(epoch):
        epoch_loss = 0
        # Training step begins
        model.train()  # Model set to training mode

        for batch_id, (x, y) in tqdm(enumerate(train_data), total=len(train_data),
                                     desc='Train epoch {epoch}'.format(epoch=i), ncols=80):
            y_ = model.forward(x.float())
            loss = criteon(y_, y)

            # Optimizer optimization model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.item()
            single_lost_list.append(loss.item())
            

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("training steps: {}, Loss: {}".format(total_train_step, loss.item()))

        
        # Testing step begins
        model.eval()  # Model set to evaluation mode
        test_loss = 0
        total_used_resources = 0  # The total number of resources used by the user
        Hit = 0
        NDCG = 0
        total_test_num = 0

        k = 10

        with torch.no_grad():
            for batch_id, (x, y) in tqdm(enumerate(test_data), total=len(test_data),
                                         desc='Test epoch {epoch}'.format(epoch=i), ncols=80):
                y_ = model.forward(x.float())
                bt_loss = criteon(y_, y)
                test_loss = test_loss + bt_loss.item()

                b_hit = 0
                b_ndcg = 0

                for j in range(len(y)):
                    total_test_num = total_test_num + 1
                    single_rel = []
                    total_used_resources = total_used_resources + len(torch.nonzero(y[j]))

                    a = torch.topk(y_[j], k).indices
                    a = a.numpy().tolist()
                    a = list(a)
               
                    b = torch.nonzero(y[j]).squeeze()
                    b = b.numpy().tolist()
                    if isinstance(b, int):
                        b = [b]
                
                    if set(a) & set(b):
                        Hit = Hit + 1

                    for item in torch.topk(y_[j], k).indices:  # The indices here are a built-in value of the torch.topk function
                        if item in torch.nonzero(y[j]):  # Note: torch.nonzero() returns the index of a non-zero position, which is valid in one dimension and different in high dimensions.
                            b_hit = b_hit + 1
                            single_rel.append(1)
                        else:
                            single_rel.append(0)

                    dcg = 0
                    idcg = 0

                    for c in range(k):
                        dcg = dcg + (single_rel[c] / numpy.log2(c+2))

                    single_rel.sort(reverse=True)  # Sort by rel from largest to smallest

                    for q in range(k):
                        idcg = idcg + (single_rel[q] / numpy.log2(q+2))

                    if idcg != 0:
                        ndcg = dcg / idcg
                    else:
                        ndcg = 0

                    b_ndcg = b_ndcg + ndcg

                Hit = Hit + b_hit
                NDCG = NDCG + b_ndcg


        epoch_loss_list.append(epoch_loss)
  
        print("Epoch HR:{}".format(Hit/total_test_num))
        print("Epoch NDCG:{}".format(NDCG/total_test_num))
        hr_list.append(Hit/total_test_num)
        ndcg_list.append(NDCG/total_test_num)
        hr_ndcg_dict[Hit/total_test_num] = NDCG/total_test_num
        
    print('The best epoch HR@10 value: {} and NDCG@10 value: {}'.format(max(hr_list), hr_ndcg_dict[max(hr_list)]))
    
    with open(r'./data/mooc_mlp_loss10.txt', 'w', encoding='utf-8') as file:
            for loss in epoch_loss_list:
                file.write(str(loss)+'\n')
                
    with open(r'./data/mooc_mlp_ndcg10.txt', 'w', encoding='utf-8') as file_2:
            for ndcg in ndcg_list:
                file_2.write(str(ndcg)+'\n')
                
    with open(r'./data/mooc_mlp_hr10.txt', 'w', encoding='utf-8') as file_3:
            for hr in hr_list:
                file_3.write(str(hr)+'\n')
    
    plt.figure()
    plt.plot(single_lost_list)
    plt.xlabel('Train step')
    plt.ylabel('Loss')
    plt.savefig(r'./data/mooc_mlp_loss_10.png')
    
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2,1,1)
    plt.plot(hr_list)
    plt.xlabel('Epoch')
    plt.ylabel('HR@10')
    plt.subplot(2,1,2)
    plt.plot(ndcg_list)
    plt.xlabel('Epoch')
    plt.ylabel('NDCG@10')
    plt.savefig(r'./data/mooc_mlp_hr_ndcg_10.png')


if __name__ == '__main__':
    train_data, test_data, train_dataset = dataiter()
    train(train_data, test_data, train_dataset)

