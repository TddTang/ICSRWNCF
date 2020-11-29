import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_one(nn.Module):
    def __init__(self, n_input, n_output):
        super(Model_one, self).__init__()
        self.predict = nn.Linear(n_input, n_output, bias=False)

    def forward(self, x):
        out = self.predict(x)
        return out


class Model_two(nn.Module):
    def __init__(self, n_input, n_output):
        super(Model_two, self).__init__()
        self.hidden1 = nn.Linear(n_input, 128)
        self.dropout1 = nn.Dropout(p=0.4)
        self.hidden2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.4)
        self.hidden3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(p=0.4)
        self.predict = nn.Linear(32, n_output)
        self.dropout4 = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.hidden1(x)
        out = F.relu(out)

        out = self.hidden2(out)
        out = F.relu(out)
        out = self.hidden3(out)
        out = F.relu(out)
        predict = self.predict(out)
        predict = torch.sigmoid(predict)
        return predict


class Model(nn.Module):
    def __init__(self, category_len, grid_len, K):
        super(Model, self).__init__()
        self.category_model = Model_one(category_len, K)
        self.grid_model = Model_one(grid_len, K)
        self.category_bias_model = Model_one(category_len, K)
        self.grid_bias_model = Model_two(grid_len, K)
        self.MLP_model = Model_two(2 * K, K)
        self.NeuMF_model = Model_one(2 * K, 1)

    def forward(self, category_input, grid_input):
        category_out = self.category_model(category_input)
        grid_out = self.grid_model(grid_input)
        SVD_out = category_out * grid_out
        bu = self.category_bias_model(category_input)
        bi = self.grid_bias_model(grid_input)
        SVD_out = SVD_out + bu + bi
        MLP_input = torch.cat((category_out, grid_out), 1)  # 横向拼接
        MLP_out = self.MLP_model(MLP_input)
        NeuMF_RS_input = torch.cat((SVD_out, MLP_out), 1)  # 横向拼接
        NeuMF_RS_out = self.NeuMF_model(NeuMF_RS_input)

        return NeuMF_RS_out


if __name__ == '__main__':
    model = Model(10, 15, 32)
    model_x = Model_one(10, 32)
    model_y = Model_one(15, 32)
    model_z = Model_two(64, 32)
    x = torch.randn(10, 10)
    y = torch.randn(10, 15)
    z = torch.randn(10, 1)
    # z_input = torch.randn(10, 64)
    # print("torch.randn(10, 64) :")
    # print(z_input.shape)
    # model_z(z_input)
    # print("--------success---------")
    # x_out = model_x(x)
    # y_out = model_y(y)
    # print(x_out.shape)
    # print(y_out.shape)
    # z_input = torch.cat((x_out, y_out), 1)  # 横向拼接
    # print("z_input = torch.cat((x_out, y_out), 1) :")
    # print(z_input.shape)
    # model_z(z_input)
    print("x: " + str(x.shape))
    print("y: " + str(y.shape))
    print("z: " + str(z.shape))
    out = model(x, y)
    print("model_out: ", out.shape)
    print(out)
    # print(model)
