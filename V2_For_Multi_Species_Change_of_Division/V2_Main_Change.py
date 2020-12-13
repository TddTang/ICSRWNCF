import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from time import time
from V2_For_Multi_Species_Change_of_Division.Information import Information
from V2_For_Multi_Species_Change_of_Division.DataLoader import DataLoader
from V2_For_Multi_Species_Change_of_Division.Net import Model
from V2_For_Multi_Species_Change_of_Division.Metrics import ndcg_at_n_for_mutli, hr_at_n_for_multi

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


def get_mode_out(now_model, category_feature, grid_feature):
    out = now_model(category_feature, grid_feature)
    out = torch.sigmoid(out)  # It can fit in the model in the same way.
    return out.view(1, len(out))[0]


if __name__ == '__main__':
    system_init(981125)
    info = Information()
    hr_list = []
    ndcg_list = []

    print("-------Initialization complete-------")

    # load data
    data = DataLoader(info)
    batch_size = info.batch_size
    train_category_index = \
        [data.train_category_index[i: i + batch_size] for i in range(0, len(data.train_category_index), batch_size)]
    train_grid_index = \
        [data.train_grid_index[i: i + batch_size] for i in range(0, len(data.train_grid_index), batch_size)]
    train_real_score_index = \
        [data.train_real_score_index[i: i + batch_size] for i in range(0, len(data.train_real_score_index), batch_size)]
    print("-------Data loading and batch partitioning complete-------")

    # construct model and optimizer
    model = Model(len(data.category_one_hot_matrix), len(data.address_block_one_hot_matrix), info.K)

    optimizer = torch.optim.Adam(model.parameters(), lr=info.lr)

    print("-------Model and Optimization Methodology Build Complete--------")

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
                ndcg = ndcg_at_n_for_mutli(test_real_score, test_NeuMF_out, info.N, info.test_num + 1,
                                           len(info.similar_categories))
                hr = hr_at_n_for_multi(test_real_score, test_NeuMF_out, info.N, info.test_num + 1,
                                       len(info.similar_categories))
                hr_list.append(hr)
                ndcg_list.append(ndcg)

                if DEBUG:
                    print('Evaluate: Epoch: ', epoch, 'Time: ', time() - time_iter, 'HR: ', hr,
                          'NDCG: ', ndcg)

                # ------------train evaluate------------
                # all_train_category_feature, all_train_grid_feature, all_train_real_score = \
                #     data.get_feature(data.train_category_index, data.train_grid_index, data.train_real_score_index)
                #
                # all_train_NeuMF_out = get_mode_out(model, all_train_category_feature, all_train_grid_feature)
                #
                # all_train_ndcg = train_ndcg_at_n(all_train_real_score, all_train_NeuMF_out,  7, 810 - (info.test_num + 1))
                # all_train_hr = train_hr_at_n(all_train_real_score, all_train_NeuMF_out, 7, 810 - (info.test_num + 1))
                # print('**Train Evaluate: Epoch: ', epoch, 'Time: ', time() - time_iter,  'Train HR: ', all_train_hr,
                #       'Train NDCG: ', all_train_ndcg)
                # ----------------------------------

    print("-------Model training complete--------")

    hr_list = np.array(hr_list)
    ndcg_list = np.array(ndcg_list)

    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("HR@" + str(info.N))
    plt.plot(range(len(hr_list)), hr_list)
    plt.subplot(1, 2, 2)
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
        ndcg = ndcg_at_n_for_mutli(test_NeuMF_out, test_real_score, info.N, info.test_num + 1,
                                   len(info.similar_categories))
        hr = hr_at_n_for_multi(test_NeuMF_out, test_real_score, info.N, info.test_num + 1, len(info.similar_categories))

        if DEBUG:
            print('Final Evaluate: ', 'Mean Loss: ', loss.item() / len(test_grid_feature), 'HR: ', hr, 'NDCG: ', ndcg)
