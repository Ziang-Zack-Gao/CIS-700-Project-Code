from dt_lib import *
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--problem_name', dest='problem_name', default='DT_20',
                    help='name of the problem. CH_n, DT_n or TSP_n.')
parser.add_argument('--train_size', dest='train_size', type=int, default=1280000, help='# of training data')
parser.add_argument('--val_size', dest='val_size', type=int, default=12800, help='# of validating data')
parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=6, help='# of training epochs')
parser.add_argument('--use_sdne', dest='use_sdne', type=bool, default=True,
                    help='use sden or linear layer as graph embedding')
parser.add_argument('--use_cuda', dest='use_cuda', type=bool, default=False, help='use GPU acceleration')
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
parser.add_argument('--max_grad_norm', dest='max_grad_norm', type=float, default=100., help='maximum gradient')
parser.add_argument('--threshold', dest='threshold', type=float, default=3.49, help='the threshold of early stop')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='')
parser.add_argument('--add_noise', dest='add_noise', type=bool, default=False,
                    help='add noise to graph embedding output')


args = parser.parse_args()


def main():

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
