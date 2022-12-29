import argparse


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='Number of processes, do not set! Done by multiproc module')

    return parser


def main():

    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    with open(r'debug_' + str(args.rank) + '.txt', 'w') as fout:
        fout.write(str(args.rank))
        fout.write(str(args.world_size))


if __name__ == '__main__':
    main()
