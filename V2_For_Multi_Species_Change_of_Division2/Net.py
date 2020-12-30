
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
        self.hidden2 = nn.Linear(128, 64)
        self.predict = nn.Linear(64, n_output)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.9)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.hidden2(out)
        out = self.dropout(out)
        out = F.relu(out)

        predict = self.predict(out)
        predict = self.dropout2(predict)
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
        MLP_input = torch.cat((category_out, grid_out), 1)  # horizontal splicing
        MLP_out = self.MLP_model(MLP_input)
        NeuMF_RS_input = torch.cat((SVD_out, MLP_out), 1)
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
    print("x: " + str(x.shape))
    print("y: " + str(y.shape))
    print("z: " + str(z.shape))
    out = model(x, y)
    print("model_out: ", out.shape)
    print(out)
    # print(model)
