# Nanjing:
# one longitude = 92819.61471090886m
# one latitude = 111864.406779661m

class Information:

    def __init__(self):
        self.x1 = 118.595752  # upper left corner(x1, y2) lower right corner(x2, y1)
        self.x2 = 118.911609
        self.y1 = 31.855101
        self.y2 = 32.09624
        self.width = 1000  # width of block
        self.longitudeBase = 92819.61471090886
        self.latitudeBase = 111864.406779661
        self.data_url = "../db/nanjing.csv"
        self.test_num = 60  # Number of non-interactive tests per category, total number of tests_num+1
        self.interactive_threshold = 20  # interaction number threshold
        self.batch_size = 32
        self.K = 20
        self.lr = 0.005
        self.n_epoch = 200
        self.print_every = 1000
        self.evaluate_every = 1
        self.test_category = '火锅'
        self.similar_categories = ['川菜家常菜', '串串香', '烧烤']

        self.N = 6  # The former N's ndcg and hr
