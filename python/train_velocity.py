import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from HIGNN.Data3 import Data3
from HIGNN.Dataloader3 import DataLoader
from HIGNN.model_structure import Two_body_net, Three_body_net, Two_body_self_net, Two_body_net_Hmatrix, HIGNN_mdoel
import os
import random


# ################ Part 1: load raw_data ################
# 1.1 two sphere data
# Note: raw_data.shape is (data_number, 9).
# raw_data = [X2, V1, V2], as X1 = [0, 0, 0] is ignored.
# force is determined by how the data is generated. 
raw_data_2body = np.loadtxt('python/data/data_2cir_ub_velocity_200k.txt', dtype=np.float32)
force_2body = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)

X1 = np.zeros((raw_data_2body.shape[0], 3), dtype=np.float32)
raw_data_2body = np.concatenate((X1, raw_data_2body), axis=1)
print(raw_data_2body.shape)

# 1.2 three sphere data
# Note: raw_data.shape is (data_number, 15).
# raw_data = [X2, X3, V1, V2, V3], as X1 = [0, 0, 0] is ignored.
# force is determined by how the data is generated.
raw_data_3body = np.loadtxt('python/data/data_3cir_ub_velocity_60k.txt', dtype=np.float32)
force_3body = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)

X1 = np.zeros((raw_data_3body.shape[0], 3), dtype=np.float32)
raw_data_3body = np.concatenate((X1, raw_data_3body), axis=1)
print(raw_data_3body.shape)
print(force_3body.size())

# 1.3 Important two functions to create edges and build graph, but currently not available
# ### For large test, we need to use pybei to generate edge info, 
# ### but currently NOT available due to incompatible issue.
def build_EdgeInfo(x, three_body_cutoff, R_cut, domain):
    EdgeInfo = pybei.BodyEdgeInfo()
    EdgeInfo.setTwoBodyEpsilon(R_cut)
    EdgeInfo.setThreeBodyEpsilon(three_body_cutoff)
    EdgeInfo.setPeriodic(False)
    EdgeInfo.setDomain(domain)

    EdgeInfo.setTargetSites(x)
    EdgeInfo.buildTwoBodyEdgeInfo()
    EdgeInfo.buildThreeBodyEdgeInfo()
    return EdgeInfo


# Create Edge
# 2body edge: (j, i), attr: F_j, from j to i
# 3body edge: (j, k, i), attr: F_j, from j to k to i
# 2body_self edge: (j, i), attr: F_i, from i to j to i
def create_edge(edgeInfo, indexRange):
    twoBodyEdgeInfo = edgeInfo.getTwoBodyEdgeInfoByIndex(indexRange)
    twoBodyEdgeInfo = twoBodyEdgeInfo.astype(int)
    edge_index_2body = torch.tensor(twoBodyEdgeInfo, dtype=torch.long)
    threeBodyEdgeInfo = edgeInfo.getThreeBodyEdgeInfoByIndex(indexRange)
    threeBodyEdgeInfo = threeBodyEdgeInfo.astype(int)
    edge_index_3body = torch.tensor(threeBodyEdgeInfo, dtype=torch.long)
    threeBodyEdgeSelfInfo = edgeInfo.getThreeBodyEdgeSelfInfoByIndex(indexRange)
    threeBodyEdgeSelfInfo = threeBodyEdgeSelfInfo.astype(int)
    edge_index_self = torch.tensor(threeBodyEdgeSelfInfo, dtype=torch.long)
    return edge_index_2body, edge_index_3body, edge_index_self



# 1.4 function to transfer raw data to the data that we can use for training.
def create_data(raw_data, force, R_cutoff_3body):
    v0 = 1.0
    # v0 is the one-body velocity. It is 1 for unbounded case. For periodic case it is smaller than 1 and depends on the size of box.
    Nc = int((raw_data.shape[1] + 3) / 6)
    print(Nc)
    N_data = raw_data.shape[0]
    indexRange = np.arange(Nc)
    data_list = list()

    domain = np.array([[-10000.0, -10000.0, -10000.0], [10000.0, 10000.0, 10000.0]], dtype="float32")
    # Define the domain, here we use a box of 32.

    for i in range(N_data):
        if i % 10000 == 0:
            print(i)
        input_number = i % force.size(0)
        positions = raw_data[i, :3 * Nc].reshape((Nc, 3))
        x = torch.tensor(raw_data[i, :3 * Nc].reshape((Nc, 3)), dtype=torch.float)
        y = torch.tensor(raw_data[i, 3 * Nc:].reshape((Nc, 3)), dtype=torch.float)
        force_in = force[input_number]

        # #### This is what we want to do but can not.
        # EdgeInfo = build_EdgeInfo(positions, R_cutoff_3body, 1.0, domain)
        # edge_index_2body, edge_index_3body, edge_index_self = create_edge(EdgeInfo, indexRange)

        # #### So here, we manually define the edge info for training since it is small system.
        if Nc == 3:
            edge_index_2body = torch.tensor([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.long)
            edge_index_3body = torch.tensor([[2, 1, 0, 2, 0, 1], [1, 2, 2, 0, 1, 0], [0, 0, 1, 1, 2, 2]],
                                            dtype=torch.long)
            edge_index_self = torch.tensor([[1, 2, 2, 0, 1, 0], [0, 0, 1, 1, 2, 2]], dtype=torch.long)
        elif Nc == 2:
            edge_index_2body = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
            edge_index_3body = torch.zeros((3, 0), dtype=torch.long)
            edge_index_self = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        
        edge_attr_2body = force_in[edge_index_2body[0, :], :]
        edge_attr_3body = force_in[edge_index_3body[0, :], :]
        edge_attr_self = force_in[edge_index_self[1, :], :]

        edge_index_one = torch.arange(0, Nc, 1).reshape((1, Nc))
        edge_index_one = edge_index_one.long()
        edge_attr_one = v0 * force_in
        data = Data3(x=x, y=y, edge_index=edge_index_2body, edge_index1=edge_index_3body, edge_attr=edge_attr_2body,
                     edge_attr1=edge_attr_3body, edge_indexs=edge_index_self, edge_attrs=edge_attr_self,
                     edge_index_one=edge_index_one, edge_attr_one=edge_attr_one)
        data_list.append(data)
    return data_list


# create the list of data
data_list_2body = create_data(raw_data_2body[0:10000,:], force_2body, 10.0)
data_list_3body = create_data(raw_data_3body, force_3body, 10.0)

print(data_list_2body[0].x)
print(data_list_2body[0].y)
print(data_list_2body[0].edge_index)
print(data_list_2body[0].edge_index1)
print(data_list_2body[0].edge_indexs)
print(data_list_2body[0].edge_index_one)
print(data_list_2body[0].edge_attr)
print(data_list_2body[0].edge_attr1)
print(data_list_2body[0].edge_attrs)
print(data_list_2body[0].edge_attr_one)
data_list = data_list_2body + data_list_3body
random.shuffle(data_list)

print(len(data_list))




# ############ Part 2: training process ##########
# split for training and test
train_list = data_list[0: 10000]
test_list = data_list[10000:]
# use dataloader to get batch
train_loader = DataLoader(train_list, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_list, batch_size=2048, shuffle=True)
print(type(train_loader))

# define three MLP models for 2-body, 3-body, and 2-body self 
# 2-body self is account for the interaction from i to j to i, contribute to Mii in paper
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


nn_2body = nn.Sequential(nn.Linear(3, 256, bias=True),
                         nn.Tanh(),
                         nn.Linear(256, 1024, bias=True),
                         nn.Tanh(),
                         nn.Linear(1024, 256, bias=True),
                         nn.Tanh(),
                         # nn.Linear(512, 256, bias=True),
                         # nn.Tanh(),
                         nn.Linear(256, 9, bias=True))

nn_3body = nn.Sequential(nn.Linear(6, 256, bias=True),
                         nn.Tanh(),
                         nn.Linear(256, 1024, bias=True),
                         nn.Tanh(),
                         # nn.Linear(1024, 1024, bias=True),
                         # nn.Tanh(),
                         nn.Linear(1024, 512, bias=True),
                         nn.Tanh(),
                         nn.Linear(512, 9, bias=True))

nn_self = nn.Sequential(nn.Linear(3, 256, bias=True),
                        nn.Tanh(),
                        nn.Linear(256, 1024, bias=True),
                        nn.Tanh(),
                        # nn.Linear(1024, 1024, bias=True),
                        # nn.Tanh(),
                        nn.Linear(1024, 512, bias=True),
                        nn.Tanh(),
                        nn.Linear(512, 9, bias=True))

# Sometimes directly training 3 models can be slow and challenging.
# It is recommended to train the nn_2body first and then fix nn_2body to train the others. 
# Finally finetune three model can get a very high accuracy.

# Load some trained model to finetune
# nn_2body = torch.load("Saved_Model/Unbounded_try1/HIGNN_nn_2body.pkl", weights_only = False)
# nn_3body = torch.load('Saved_Model/Unbounded_try1/HIGNN_nn_3body.pkl', weights_only = False)
# nn_self = torch.load('Saved_Model/Unbounded_try1/HIGNN_nn_self.pkl', weights_only = False)


nn_2body = Two_body_net_Hmatrix(nn_2body)
nn_3body = Three_body_net(nn_3body)
nn_self = Two_body_self_net(nn_self)

model = HIGNN_mdoel(nn_2body=nn_2body, nn_3body=nn_3body, nn_self=nn_self).to(device)

# #### training part
# These parameters e.g. lr can be changed to train a better model.
optimizer = torch.optim.Adam(model.parameters(), lr=0.00008)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
# scheduler = MultiStepLR(optimizer, milestones=[80, 100, 120, 140, 160, 180], gamma=0.4)
MSELoss = nn.MSELoss()


def extract_info(data, device):
    x = data.x.to(device)
    edge_2body = data.edge_index.to(device)
    edge_3body = data.edge_index1.to(device)
    edge_2bodySelf = data.edge_indexs.to(device)
    edge_1body = data.edge_index_one.to(device)
    edge_attr_2body = data.edge_attr.to(device)
    edge_attr_3body = data.edge_attr1.to(device)
    edge_attr_2bodySelf = data.edge_attrs.to(device)
    edge_attr_1body = data.edge_attr_one.to(device)
    return x, edge_2body, edge_3body, edge_2bodySelf, edge_1body, edge_attr_2body, edge_attr_3body, edge_attr_2bodySelf, edge_attr_1body


def Validation_loss(loader):
    test_loss = 0.0
    num = 0
    for data in loader:
        x, edge_2body, edge_3body, edge_2bodySelf, edge_1body, edge_attr_2body, edge_attr_3body, edge_attr_2bodySelf, edge_attr_1body = extract_info(
            data, device)
        out = model(x, edge_2body, edge_3body, edge_2bodySelf, edge_1body, edge_attr_2body, edge_attr_3body,
                    edge_attr_2bodySelf, edge_attr_1body).cpu()
        test_loss += MSELoss(out, data.y - data.edge_attr_one).item()
        num += 1
    return test_loss / num


file_name = 'Unbounded_try2'
save_folder = 'Saved_Model/' + file_name 
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

file = open(save_folder + '/loss.txt', "w")

num_epochs = 450
tol_stop = 0.01

for epoch in range(num_epochs):
    train_loss_total = 0
    num = 0
    for data in train_loader:
        optimizer.zero_grad()
        x, edge_2body, edge_3body, edge_2bodySelf, edge_1body, edge_attr_2body, edge_attr_3body, edge_attr_2bodySelf, edge_attr_1body = extract_info(
            data, device)
        out = model(x, edge_2body, edge_3body, edge_2bodySelf, edge_1body, edge_attr_2body, edge_attr_3body,
                    edge_attr_2bodySelf, edge_attr_1body).cpu()
        # print(out.size())
        train_loss = MSELoss(out, data.y - data.edge_attr_one)
        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss.item()
        num += 1

    scheduler.step()
    train_loss = train_loss_total / num
    test_loss = Validation_loss(test_loader)

    if test_loss < tol_stop:
        tol_stop = test_loss
        # save model in one 
        torch.save(model, save_folder + '/HIGNN.pkl')
        # save models seperately, the HIGNN_nn_2body will be used for H-matrix.
        torch.save(model.nn_2body.net, save_folder + '/HIGNN_nn_2body.pkl')
        torch.save(model.nn_3body.net, save_folder + '/HIGNN_nn_3body.pkl')
        torch.save(model.nn_self.net, save_folder + '/HIGNN_nn_self.pkl')

    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')

    text = file.write(str(epoch) + "\t" + str(train_loss) + "\t" + str(test_loss) + "\n")

file.close()
