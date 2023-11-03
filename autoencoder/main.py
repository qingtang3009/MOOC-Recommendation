# -*- coding:utf-8 -*-
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from AE import AE
from tqdm import tqdm


def dataiter(path):
    with open(path, 'r', encoding='utf-8') as r:
        dic = json.load(r)

    i = 0
    for key, value in dic.items():
        if i <= 5:
            print(key, value)
        i = i + 1

    sequence = list(range(len(dic)))
    texts = list(dic.values())
    dataset = torch.utils.data.TensorDataset(torch.Tensor(sequence), torch.Tensor(texts))
    train_dataiter = DataLoader(dataset, batch_size=50, shuffle=True)
    dataiter = DataLoader(dataset, batch_size=1, shuffle=False)
    sample = (iter(train_dataiter).next())
    print(type(sample[1]))
    print(sample[0], sample[1])
    print(len(sample[1]))

    return train_dataiter, dataiter


def train(batch_data):
    model = AE()

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)

    model.apply(init_normal)

    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(model)

    for epoch in range(10):
        epoch_loss = 0
        for batch_id, (_, x) in tqdm(enumerate(batch_data), total=len(batch_data), desc='Epoch {epoch}'.format(epoch=epoch), ncols=80):

            code, x_ = model.forward(x)
            loss = criteon(x_, x)
            # print(code)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss

        print(epoch, 'average_loss', epoch_loss.item()/len(batch_data))

    # torch.save(model.state_dict(), './AE_model.pt')

    return model


def decoder(single_data, model, save_path):
    vecs = []
    nus = []
    with torch.no_grad():
        for batch_id, (number, x) in enumerate(single_data):
            code, x_ = model.forward(x)
            # print(number, code.squeeze())
            vecs.append((code.squeeze()).tolist())
            nus.append(number.item())

        dic = dict(zip(nus, vecs))

    json_file = json.dumps(dic, indent=4, ensure_ascii=False)
    with open(save_path, 'w', encoding='utf-8') as w:
        w.write(json_file)


if __name__ == '__main__':
    data_path = r"./data/sparse_vectors.json"
    train_dataiter, dataiter = dataiter(data_path)
    trained_model = train(train_dataiter)
    path = r"./data/ae_output.json"
    decoder(dataiter, trained_model, path)

