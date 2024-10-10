import argparse

def get_args():
    parser = argparse.ArgumentParser(description='UFG for MTS forecasting')
    # data config
    parser.add_argument(
        '--root_path', type=str, default='./data/', help='data dir'
    )
    parser.add_argument(
        '--data_name', type=str, default='Covid', help='data set',
            choices=['ECG', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Solar',
                    'Covid', 'Traffic', 'Wiki500' ]
    )
    parser.add_argument(
        '--seq_len', type=int, default=12, help='input length'
    )
    parser.add_argument(
        '--pred_len', type=int, default=12, help='prediction length',
        choices=[96, 192, 336, 720]
    )
    parser.add_argument(
        '--signal_len', type=int, default=16, help='freq length'
    )
    parser.add_argument(
        '--features', type=str, default='M', help='M\MS for MTS, S for UTS',
            choices=['M', 'MS', 'S']
    )
    parser.add_argument(
        '--target', type=str, default='OT', help='target feature in S or MS task'
    )
    # model config
    parser.add_argument(
        '--embed_size', type=int, default=128, help='embedding size of value_t'
    )
    parser.add_argument(
        '--hidden_size', type=int, default=128, help='hidden dimensions of model'
    )
    parser.add_argument(
        '--k', type=int, default=4, help='number of neighbors of KNN Graph'
    )
    parser.add_argument(
        '--cheb_order', type=int, default=2,
            help='cheb_order - 1 = Order of Chebyshev Polynomial Approximation'
    )
    parser.add_argument(
        '--lev', type=int, default=1, help='level of framelet transform'
    )
    parser.add_argument(
        '--s', type=float, default=7.533034289466433, help='framelet dilation scale (> 1)'
    )
    parser.add_argument(
        '--frame_type', type=str, default='Haar', help='frame type',
            choices=['Haar', 'Linear']
    )
    # training config
    parser.add_argument(
        '--train_epochs', type=int, default=100, help='train epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='input data batch size'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.003971709943520522, help='learning rate'
    )
    parser.add_argument(
        '--decay_rate', type=float, default=0.04883689490804201
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='device'
    )
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed'
    )
    parser.add_argument(
        '--use_compile', action='store_true', help='if use torch.compile model'
    )
    parser.add_argument(
        '--long_pred', type=int, default=0, help='pred long use 1 else 0'
    )

    args = parser.parse_args()


    return args