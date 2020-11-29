import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from V3.Information import Information
from V3.DataLoader import DataLoader
from V3.Net import Model
from V3.metrics import ndcg_at_n, hr_at_n, train_hr_at_n, train_ndcg_at_n

DEBUG = True
CUDA_AVAILABLE = False
DEVICE = None
N_GPU = 0


def system_init(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA
    global CUDA_AVAILABLE, DEVICE, N_GPU
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    N_GPU = torch.cuda.device_count()
    if N_GPU > 0:
        torch.cuda.manual_seed_all(seed)


def get_mode_out(model, category_feature, grid_feature):
    out = model(category_feature, grid_feature)
    out = torch.sigmoid(out)  # 应该可以放入模型中同理
    return out.view(1, len(out))[0]


if __name__ == '__main__':
    # init
    system_init(981125)
    info = Information()
    loss_list = []
    hr_list = []
    ndcg_list = []

    print("-------初始化完成-------")

    # load data
    data = DataLoader(info)
    batch_size = info.batch_size
    train_category_index = \
        [data.train_category_index[i: i + batch_size] for i in range(0, len(data.train_category_index), batch_size)]
    train_grid_index = \
        [data.train_grid_index[i: i + batch_size] for i in range(0, len(data.train_grid_index), batch_size)]
    train_real_score_index = \
        [data.train_real_score_index[i: i + batch_size] for i in range(0, len(data.train_real_score_index), batch_size)]
    print("-------数据装载&&batch划分完成-------")

    # construct model and optimizer
    model = Model(len(data.category_one_hot_matrix), len(data.address_block_one_hot_matrix), info.K)

    optimizer = torch.optim.Adam(model.parameters(), lr=info.lr)

    print("-------模型&优化方式构建完成--------")

    # move to GPU
    if CUDA_AVAILABLE:
        model.to(DEVICE)
        data.address_block_one_hot_matrix = data.address_block_one_hot_matrix.to(DEVICE)
        data.category_one_hot_matrix = data.category_one_hot_matrix.to(DEVICE)
        data.interaction_line = data.interaction_line.to(DEVICE)

    # training
    for epoch in range(info.n_epoch):
        model.train()
        iter_total_loss = 0
        time_iter = time()
        for batch_iter in range(len(train_category_index)):
            time_iter = time()

            optimizer.zero_grad()
            batch_total_loss = 0

            batch_train_category_index = train_category_index[batch_iter]
            batch_train_grid_index = train_grid_index[batch_iter]
            batch_train_real_score_index = train_real_score_index[batch_iter]

            batch_train_category_feature, batch_train_grid_feature, batch_train_real_score = \
                data.get_feature(batch_train_category_index, batch_train_grid_index, batch_train_real_score_index)

            NeuMF_out = get_mode_out(model, batch_train_category_feature, batch_train_grid_feature)

            batch_total_loss += \
                torch.nn.functional.binary_cross_entropy(NeuMF_out, batch_train_real_score, reduction='sum')

            # calculate total loss and backward
            iter_total_loss += batch_total_loss.item()
            batch_total_loss.backward()
            optimizer.step()

            if DEBUG and (batch_iter % info.print_every) == 0:
                print('Epoch: ', epoch, '| Batch_iter: ', batch_iter, '| Time: ', time() - time_iter, '| Iter Loss: ',
                      batch_total_loss.item(), '| Iter Mean Loss: ', iter_total_loss / (batch_iter + 1))

        # evaluate prediction model
        if (epoch % info.evaluate_every) == 0:
            model.eval()
            with torch.no_grad():
                # evaluate loss
                test_category_feature, test_grid_feature, test_real_score = \
                    data.get_feature(data.test_category_index, data.test_grid_index, data.test_real_score_index)

                test_NeuMF_out = get_mode_out(model, test_category_feature, test_grid_feature)

                loss = torch.nn.functional.binary_cross_entropy(test_NeuMF_out, test_real_score, reduction='sum')

                # evaluate ndcg & hr
                ndcg = ndcg_at_n(test_real_score, test_NeuMF_out, info.N, info.test_num + 1)
                hr = hr_at_n(test_real_score, test_NeuMF_out,  info.N, info.test_num + 1)

                # ------------train evaluate------------
                all_train_category_feature, all_train_grid_feature, all_train_real_score = \
                    data.get_feature(data.train_category_index, data.train_grid_index, data.train_real_score_index)

                all_train_NeuMF_out = get_mode_out(model, all_train_category_feature, all_train_grid_feature)

                all_train_ndcg = train_ndcg_at_n(all_train_real_score, all_train_NeuMF_out,  8, 810 - (info.test_num + 1))
                all_train_hr = train_hr_at_n(all_train_real_score, all_train_NeuMF_out, 8, 810 - (info.test_num + 1))
                print('**Train Evaluate: Epoch: ', epoch, 'Time: ', time() - time_iter,  'Train HR: ', all_train_hr,
                      'Train NDCG: ', all_train_ndcg)
                # ----------------------------------

                loss_list.append(loss)
                hr_list.append(hr)
                ndcg_list.append(ndcg)

                if DEBUG:
                    print('Evaluate: Epoch: ', epoch, 'Time: ', time() - time_iter, 'Test Loss: ', loss, 'HR: ', hr,
                          'NDCG: ', ndcg)
    print("-------模型训练完成--------")

    loss_list = np.array(loss_list)
    hr_list = np.array(hr_list)
    ndcg_list = np.array(ndcg_list)

    plt.subplot(1, 3, 1)
    plt.xlabel("epoch")
    plt.ylabel("Test Loss")
    plt.plot(range(len(loss_list)), loss_list)
    plt.subplot(1, 3, 2)
    plt.xlabel("epoch")
    plt.ylabel("HR@" + str(info.N))
    plt.plot(range(len(hr_list)), hr_list)
    plt.subplot(1, 3, 3)
    plt.xlabel("epoch")
    plt.ylabel("NDCG@" + str(info.N))
    plt.plot(range(len(ndcg_list)), ndcg_list)
    plt.tight_layout()
    plt.savefig('result.png')
    plt.show()

    # testing
    model.eval()
    with torch.no_grad():
        # evaluate loss
        test_category_feature, test_grid_feature, test_real_score = \
            data.get_feature(data.test_category_index, data.test_grid_index, data.test_real_score_index)

        test_NeuMF_out = get_mode_out(model, test_category_feature, test_grid_feature)

        loss = torch.nn.functional.binary_cross_entropy(test_NeuMF_out, test_real_score, reduction='sum')

        # evaluate ndcg & hr
        ndcg = ndcg_at_n(test_NeuMF_out, test_real_score, info.N, info.test_num + 1)
        hr = hr_at_n(test_NeuMF_out, test_real_score, info.N, info.test_num + 1)

        if DEBUG:
            print('Final Evaluate: ', 'Mean Loss: ', loss.item() / len(test_grid_feature), 'HR: ', hr, 'NDCG: ', ndcg)
