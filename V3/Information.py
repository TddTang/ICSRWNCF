# 定义所有的信息
# 南京：
# 1经度 = 92819.61471090886m
# 1维度 = 111864.406779661m
# 经度范围，维度范围，地址块宽度，当前地区1经度米数，1维度米数

class Information:

    def __init__(self):
        self.x1 = 118.595752  # 左上角(x1, y2) 右下角(x2,y1)
        self.x2 = 118.911609
        self.y1 = 31.855101
        self.y2 = 32.09624
        self.width = 1000
        self.longitudeBase = 92819.61471090886
        self.latitudeBase = 111864.406779661
        self.data_url = "../db/nanjing.csv"
        self.test_num = 60  # 每个类别无交互的测试数量，总个数为test_num+1
        self.interactive_threshold = 20
        self.batch_size = 32
        self.K = 40
        self.lr = 0.001
        self.n_epoch = 200  # 本地demo
        self.print_every = 1000
        self.evaluate_every = 1

        self.N = 8  # 前N个的ndcg和hr

        # self.x1 = 118.597718  # 左上角(x1, y2) 右下角(x2,y1)
        # self.x2 = 118.990666
        # self.y1 = 31.79446
        # self.y2 = 32.215725
        # self.width = 1000
        # self.longitudeBase = 92819.61471090886
        # self.latitudeBase = 111864.406779661
        # self.data_url = "../db/nanjing.csv"
        # self.test_num = 100  # 每个类别无交互的测试数量，总个数为test_num+1
        # self.interactive_threshold = 15
        # self.batch_size = 32
        # self.K = 40
        # self.lr = 0.001
        # self.n_epoch = 2  # 本地demo
        # self.print_every = 10
        # self.evaluate_every = 1
        #
        # self.N = 6  # 前N个的ndcg和hr