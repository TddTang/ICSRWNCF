import pandas as pd
import numpy as np
import math
import random
import torch

from V1.Information import Information


class DataLoader:

    def __init__(self, info):
        self.info = info

        self.interaction_line, self.address_block_one_hot_matrix, self.category_one_hot_matrix, \
        self.interaction_matrix_id, self.id_category, self.address_block_scope, \
        self.test_set, self.category_vis = self.distribute_data(info.data_url, info.test_num)

        self.train_category_index, self.train_grid_index, self.train_real_score_index, self.test_category_index, \
        self.test_grid_index, self.test_real_score_index = self.get_index()

    def show(self):
        print("longitude range: " + str(self.x1) + "-" + str(self.x2))
        print("latitude range" + str(self.y1) + "-" + str(self.y2))
        print("one longitude= " + str(self.longitudeBase) + "m")
        print("one latitude= " + str(self.latitudeBase) + "m")
        print("block width" + str(self.width) + "m")

    '''
        Iterate the store data twice.
        First time: counting and numbering store types
        Second time: construct a matrix of store types and address blocks that can be used to calculate the address block
        from the store's latitude and longitude and the address block number.
        
        Get the one-hot matrix of locations and the one-hot matrix of categories.
        Get a random test set
    '''
    def distribute_data(self, url, test_num):
        print("distribute data start...")
        x1 = self.info.x1
        x2 = self.info.x2
        y1 = self.info.y1
        y2 = self.info.y2
        width = self.info.width
        longitude_base = self.info.longitudeBase
        latitude_base = self.info.latitudeBase

        x_len = (x2 - x1) * longitude_base
        y_len = (y2 - y1) * latitude_base
        x_num = math.ceil(x_len / width)
        y_num = math.ceil(y_len / width)
        x_degree = width / longitude_base
        y_degree = width / latitude_base
        address_block_scope = {}
        for i in range(x_num):
            for j in range(y_num):
                square_id = i * x_num + j
                address_block_scope[square_id] = [x1 + i * x_degree, x1 + (i + 1) * x_degree,
                                                  y2 - (i + 1) * y_degree, y2 - i * y_degree]

        my_data = pd.read_csv(url, low_memory=False)
        category_num = 0
        category_id = {}  # category->id
        id_category = {}  # id->category

        num = 0
        for index, row in my_data.iterrows():
            category = row["small_category"]
            longitude = row["longitude"]
            latitude = row["latitude"]

            num += 1
            if num % 100000 == 0:
                print("Already visited " + str(num))

            if longitude < x1 or longitude > x2 or latitude < y1 or latitude > y2:
                continue
            if category not in category_id:
                category_id[category] = category_num
                id_category[category_num] = category
                category_num += 1

        print("---category num: " + str(category_num) + " ; grid num: " + str(x_num * y_num) + "---")
        num = 0
        interaction_matrix = np.full((category_num, x_num * y_num), 0, dtype=int)  # 下标都从0开始
        for index, row in my_data.iterrows():
            longitude = row["longitude"]
            latitude = row["latitude"]

            num += 1
            if num % 100000 == 0:
                print("Already visited " + str(num))

            if longitude < x1 or longitude > x2 or latitude < y1 or latitude > y2:
                continue
            else:
                now_category_id = category_id[row["small_category"]]
                x_difference = longitude - x1
                y_difference = y2 - latitude
                now_xid = math.floor(x_difference / x_degree)
                now_yid = math.floor(y_difference / y_degree)
                square_id = now_yid * x_num + now_xid

                interaction_matrix[now_category_id][square_id] = 1

        interaction_line = []
        interaction_matrix_id = np.full((category_num, x_num * y_num), 0, dtype=int)
        for i in range(category_num):
            for j in range(x_num * y_num):
                interaction_line.append([interaction_matrix[i][j]])
                interaction_matrix_id[i][j] = (len(interaction_line) - 1)

        address_block_one_hot_matrix = np.full((x_num * y_num, x_num * y_num), 0, dtype=int)
        category_one_hot_matrix = np.full((category_num, category_num), 0, dtype=int)
        for i in range(x_num * y_num):
            address_block_one_hot_matrix[i][i] = 1
        for i in range(category_num):
            category_one_hot_matrix[i][i] = 1

        test_set = []
        category_vis = []
        for i in range(category_num):
            one = []
            no_one = []
            for j in range(x_num * y_num):
                if interaction_matrix[i][j] == 1:
                    one.append(j)
                else:
                    no_one.append(j)
            if len(one) < self.info.interactive_threshold:
                category_num -= 1
                category_vis.append(0)
                test_set.append([-1])
            else:
                category_vis.append(1)
                tmp = random.sample(one, 1)
                tmp.extend(random.sample(no_one, test_num))
                test_set.append(tmp)

        print("After clear---category num: " + str(category_num) + " ; grid num: " + str(x_num * y_num) + "---")

        return torch.Tensor(interaction_line), torch.Tensor(address_block_one_hot_matrix), \
               torch.Tensor(category_one_hot_matrix), interaction_matrix_id, id_category, \
               address_block_scope, test_set, category_vis

    # Combined index + true value for constructing address blocks and categories of training and
    # combined index + true value for testing.
    def get_index(self):

        category_num = len(self.id_category)
        grid_num = len(self.address_block_scope)
        train_category_index = []
        train_grid_index = []
        train_real_score_index = []
        test_category_index = []
        test_grid_index = []
        test_real_score_index = []

        for i in range(category_num):
            if self.category_vis[i] == 0:  # invalid category (number of interactions less than the threshold)
                continue
            for j in range(grid_num):
                if j not in self.test_set[i]:
                    train_category_index.append(i)
                    train_grid_index.append(j)
                    train_real_score_index.append(self.interaction_matrix_id[i][j])
                else:
                    test_category_index.append(i)
                    test_grid_index.append(j)
                    test_real_score_index.append(self.interaction_matrix_id[i][j])

        return train_category_index, train_grid_index, train_real_score_index, test_category_index, \
               test_grid_index, test_real_score_index

    # The corresponding vector is retrieved from the matrix according to the index, and the type returned is the tensor.
    def get_feature(self, category_index_set, grid_index_set, real_score_set):
        now_real_score = self.interaction_line[real_score_set]
        now_real_score = now_real_score.view(1, len(now_real_score))[0]

        return self.category_one_hot_matrix[category_index_set], \
               self.address_block_one_hot_matrix[grid_index_set], now_real_score


if __name__ == '__main__':
    my_info = Information()
    myDateLoader = DataLoader(my_info)
    category_feature, grid_feature, real_score = myDateLoader.get_feature(myDateLoader.test_category_index,
                                                                          myDateLoader.test_grid_index,
                                                                          myDateLoader.test_real_score_index)
    print(type(category_feature))
    print(type(grid_feature))
    print(type(real_score))
    print("run over")
