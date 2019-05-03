from tsp_lib import *
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--tutorial', dest='tutorial', type=bool, default=False, help='want tutorial or not')
parser.add_argument('--train_tsp_20', dest='train_tsp_20', type=bool, default=True, help='')
parser.add_argument('--train_tsp_50', dest='train_tsp_50', type=bool, default=True, help='')
parser.add_argument('--problem_name', dest='problem_name', default='TSP_20',
                    help='name of the problem. CH_n, DT_n or TSP_n.')
parser.add_argument('--train_size', dest='train_size', type=int, default=1280000, help='# of training data')
parser.add_argument('--val_size', dest='val_size', type=int, default=12800, help='# of validating data')
parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=6, help='# of training epochs')
parser.add_argument('--use_sdne', dest='use_sdne', type=bool, default=True,
                    help='use sden or linear layer as graph embedding')
parser.add_argument('--use_cuda', dest='use_cuda', type=bool, default=True, help='use GPU acceleration')
parser.add_argument('--attention', dest='attention', default='Bahdanau', choices=['Dot', 'Bahdanau'],
                    help='attention mechanism')
parser.add_argument('--ebd_size', dest='ebd_size', type=int, default=128, help='the size of embedded vector')
parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=128, help='# of hidden units')
parser.add_argument('--n_glimpses', dest='n_glimpses', type=int, default=1, help='# of glimpses')
parser.add_argument('--tanh_exploration', dest='tanh_exploration', type=int, default=10, help='')
parser.add_argument('--use_tanh', dest='use_tanh', type=bool, default=True, help='use tanh as activate function')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.99999, help='the decay rate')
parser.add_argument('--use_decay', dest='use_decay', type=bool, default=False,
                    help='use the decay of critic\'s expectation')
parser.add_argument('--beta', dest='beta', type=float, default=0.9,
                    help='the base of the exponential moving of the critic\'s expectation')
parser.add_argument('--max_grad_norm', dest='max_grad_norm', type=float, default=2., help='maximum gradient')
parser.add_argument('--threshold', dest='threshold', type=float, default=3.49, help='the threshold of early stop')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='')
parser.add_argument('--add_noise', dest='add_noise', type=bool, default=False,
                    help='add noise to graph embedding output')


args = parser.parse_args()


def tutorial(train_tsp_20=True, train_tsp_50=True):
    train_size = 1000000
    val_size = 10000

    use_sdne = True
    ebd_size = 128
    hidden_size = 128
    n_glimpses = 1
    tanh_exploration = 10
    use_tanh = True
    USE_CUDA = True

    time_1 = time.time()
    if train_tsp_20:
        train_20_dataset = DatasetGenerator(20, train_size, name='TSP_20')
        val_20_dataset = DatasetGenerator(20, val_size, name='TSP_20')
    if train_tsp_50:
        train_50_dataset = DatasetGenerator(50, train_size, name='TSP_50')
        val_50_dataset = DatasetGenerator(50, val_size, name='TSP_50')
    time_2 = time.time()
    print("Time cost for generating datasets: %4.4f s" % (time_2 - time_1))

    if train_tsp_20:
        tsp_20_model = CombinatorialRL(
            ebd_size,
            hidden_size,
            20,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            reward,
            attention="Dot",
            use_cuda=USE_CUDA,
            use_sdne=use_sdne,
            name='TSP_20')

    if train_tsp_50:
        tsp_50_model = CombinatorialRL(
            ebd_size,
            hidden_size,
            50,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            reward,
            attention="Bahdanau",
            use_cuda=USE_CUDA,
            use_sdne=use_sdne,
            name='TSP_50')

    if USE_CUDA:
        if train_tsp_20:
            tsp_20_model = tsp_20_model.cuda()
        if train_tsp_50:
            tsp_50_model = tsp_50_model.cuda()

    if train_tsp_20:
        print("\nReady to train TSP 20...")
        tsp_20_train = TrainModel(args,
                                  tsp_20_model,
                                  train_20_dataset,
                                  val_20_dataset,
                                  # original threshold = 3.99
                                  threshold=3.49,
                                  continue_train=True,
                                  name='TSP_20')
        tsp_20_train.train_and_validate(args, 10)

    if train_tsp_50:
        print("\nReady to train TSP 50...")
        train_50_train = TrainModel(args,
                                    tsp_50_model,
                                    train_50_dataset,
                                    val_50_dataset,
                                    # original threshold = 6.4
                                    threshold=6.,
                                    continue_train=False,
                                    name='TSP_50')
        train_50_train.train_and_validate(args, 10)


def main():
    if args.tutorial:
        tutorial(train_tsp_20=args.train_tsp_20, train_tsp_50=args.train_tsp_50)

    time_1 = time.time()
    idx = args.problem_name.index('_')
    num_nodes = int(args.problem_name[idx+1:])
    train_dataset = DatasetGenerator(num_nodes, args.train_size, name=args.problem_name)
    val_dataset = DatasetGenerator(num_nodes, args.val_size, name=args.problem_name)

    time_2 = time.time()
    print("Time cost for generating datasets: %4.4f s" % (time_2 - time_1))

    model = CombinatorialRL(
        args.ebd_size,
        args.hidden_size,
        num_nodes,
        args.n_glimpses,
        args.tanh_exploration,
        args.use_tanh,
        reward,
        attention=args.attention,
        use_cuda=args.use_cuda,
        use_sdne=args.use_sdne,
        name=args.problem_name)

    if args.use_cuda:
        model.cuda()

    print("\nReady to train %s..." % args.problem_name)
    train = TrainModel(args,
                       model,
                       train_dataset,
                       val_dataset,
                       threshold=args.threshold,
                       continue_train=args.continue_train,
                       max_grad_norm=args.max_grad_norm,
                       name=args.problem_name)
    train.train_and_validate(args, args.n_epoch)
    print("\nTraining ends!")


if __name__ == '__main__':
    main()
