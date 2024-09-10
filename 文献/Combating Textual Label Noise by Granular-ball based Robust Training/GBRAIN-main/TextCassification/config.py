import argparse  # argparse 模块，用于解析命令行参数。
parser = argparse.ArgumentParser(description='Train Text Classification')
parser.add_argument('--data_type', default='agnews', type=str,
                    choices=['imdb', 'newsgroups', 'reuters', 'webkb', 'cade', 'dbpedia', 'agnews', 'yahoo','sogou', 'yelp', 'amazon'], 
                    help='dataset type')
parser.add_argument('--fine_grained', action='store_true', help='use fine grained class or not, it only works for '
                                                                'reuters, yelp and amazon')
parser.add_argument('--text_length', default=300, type=int, help='the number of words about the text to load')

parser.add_argument('--print_intervals', default=20, type=int, help='print_intervals')
# parser.add_argument('--routing_type', default='k_means', type=str, choices=['k_means', 'dynamic'],
#                     help='routing type, it only works for capsule classifier')
parser.add_argument('--routing_type', default='dynamic', type=str, choices=['k_means', 'dynamic'],
                    help='routing type, it only works for capsule classifier')


parser.add_argument('--loss_type', default='mf', type=str,
                    choices=['margin', 'focal','GCE', 'cross', 'mf', 'mc', 'fc', 'mfc'], help='loss type')
parser.add_argument('--embedding_type', default='normal', type=str, choices=['cwc', 'cc', 'normal'],
                    help='embedding type')
# parser.add_argument('--classifier_type', default='capsule', type=str, choices=['capsule', 'linear'],
#                     help='classifier type')
parser.add_argument('--classifier_type', default='linear', type=str, choices=['capsule', 'linear'],
                    help='classifier type')
parser.add_argument('--embedding_size', default=100, type=int, help='embedding size')
parser.add_argument('--num_codebook', default=8, type=int,
                    help='codebook number, it only works for cwc and cc embedding')
parser.add_argument('--num_codeword', default=None, type=int,
                    help='codeword number, it only works for cwc and cc embedding')
parser.add_argument('--hidden_size', default=150, type=int, help='hidden size')
parser.add_argument('--in_length', default=8, type=int,
                    help='in capsule length, it only works for capsule classifier')
parser.add_argument('--out_length', default=16, type=int,
                    help='out capsule length, it only works for capsule classifier')
parser.add_argument('--num_iterations', default=3, type=int,
                    help='routing iterations number, it only works for capsule classifier')
parser.add_argument('--num_repeat', default=10, type=int,
                    help='gumbel softmax repeat number, it only works for cc embedding')
parser.add_argument('--drop_out', default=0.5, type=float, help='drop_out rate of GRU layer')
parser.add_argument('--round', default=1, type=int, help='重复次数')
parser.add_argument('--purity', default=0.6, type=float, help='粒球纯度')
parser.add_argument('--recluster',default=3,type=int,help='聚球次数')
parser.add_argument('--drop_ball',default=0.4,type=float,help='保留的球')
parser.add_argument('--noise_p',default=0.3,type=float,help='噪声比例')
parser.add_argument('--min_ball',default=3,type=int,help='最小要保留的球')

parser.add_argument('--chunk_ball',default=5,type=int,help='第二轮扔掉的球')



parser.add_argument('--batch_size', default=256, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=20, type=int, help='train epochs number')
parser.add_argument('--num_steps', default=100, type=int, help='test steps number')
parser.add_argument('--pre_model', default=None, type=str,
                    help='pre-trained model weight, it only works for routing_type experiment')
opt = parser.parse_args()

