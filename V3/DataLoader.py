import pandas as pd
import numpy as np
import math
import random
import csv
import torch

from V3.Information import Information

# 南京：
# 1经度 = 92819.61471090886m
# 1维度 = 111864.406779661m


class DataLoader:

    def __init__(self, info):
        self.info = info

        self.interaction_line, self.address_block_one_hot_matrix, self.category_one_hot_matrix, \
        self.interaction_matrix_id, self.id_category, self.address_block_scope, \
        self.test_set, self.category_vis = self.distribute_data(info.data_url, info.test_num)

        # interaction_line, address_block_one_hot_matrix, category_one_hot_matrix 为tensor类型
        # print("[1/...] 训练的交互矩阵，地址块one-hot矩阵，类别one-hot矩阵，类别序号对类别的映射，地址块序号对应的范围，测试集合")

        self.train_category_index, self.train_grid_index, self.train_real_score_index, self.test_category_index, \
        self.test_grid_index, self.test_real_score_index = self.get_index()
        # score 为交互矩阵中真实值 为tensor类型

        # print("[2/...] 地址块和类别块训练的组合索引+真实值 和 测试的组合索引+真实值")

    def show(self):
        print("经度范围：" + str(self.x1) + "-" + str(self.x2))
        print("维度范围：" + str(self.y1) + "-" + str(self.y2))
        print("当前地区1经度= " + str(self.longitudeBase) + "m")
        print("当前地区1维度= " + str(self.latitudeBase) + "m")
        print("地址块宽度：" + str(self.width) + "m")

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
        # 遍历两次店铺数据：
        # 第一次：统计店铺类型的数量并对店铺类型进行编号
        # 第二次：构建店铺类型和地址块的矩阵，可以由店铺所处的经纬度算出店铺所在的地址块并算的改地址块的编号
        # 得到地点的one-hot矩阵和类别的one-hot矩阵
        # 随机得到测试集合
        my_data = pd.read_csv(url, low_memory=False)
        category_num = 0
        category_id = {}  # 类型->id
        id_category = {}  # id->类型

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

        # print("[1/4] distribute data")
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

        # print("[2/4] distribute data")

        address_block_one_hot_matrix = np.full((x_num * y_num, x_num * y_num), 0, dtype=int)
        category_one_hot_matrix = np.full((category_num, category_num), 0, dtype=int)
        for i in range(x_num * y_num):
            address_block_one_hot_matrix[i][i] = 1
        for i in range(category_num):
            category_one_hot_matrix[i][i] = 1

        # print("[3/4] distribute data")

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
        # print("[4/4] distribute data")

        # 返回训练的交互矩阵列，地址块one-hot矩阵，类别one-hot矩阵，交互矩阵id号，类别序号对类别的映射，地址块序号对应的范围，测试集合
        return torch.Tensor(interaction_line), torch.Tensor(address_block_one_hot_matrix), \
               torch.Tensor(category_one_hot_matrix), interaction_matrix_id, id_category, \
               address_block_scope, test_set, category_vis

    # 构造地址块和类别块训练的组合索引+真实值 和 测试的组合索引+真实值
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
            if self.category_vis[i] == 0:  # 无效类别 (交互次数少于阈值
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

    # 根据index，在矩阵中取出对应的向量,返回为tensor
    def get_feature(self, category_index_set, grid_index_set, real_score_set):
        now_real_score = self.interaction_line[real_score_set]
        now_real_score = now_real_score.view(1, len(now_real_score))[0]

        return self.category_one_hot_matrix[category_index_set], \
               self.address_block_one_hot_matrix[grid_index_set], now_real_score


if __name__ == '__main__':
    my_info = Information()  # 信息都包含在info中
    myDateLoader = DataLoader(my_info)
    category_feature, grid_feature, real_score = myDateLoader.get_feature(myDateLoader.test_category_index,
                                                                          myDateLoader.test_grid_index,
                                                                          myDateLoader.test_real_score_index)
    print(type(category_feature))
    print(type(grid_feature))
    print(type(real_score))
    print("run over")
