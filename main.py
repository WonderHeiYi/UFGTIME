import time
import argparse

import torch
import torch.nn as nn
import numpy as np
import optuna
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.model import FreqTimeUFGV2, FGN, Model
from src.utils import evaluate, set_seed, get_frame, cheb_approx
from src.data_provider import data_provider
from argument import get_args
import pandas as pd

if __name__ == '__main__':

    args = get_args()
    # def run_tune(trial):
    #     args = args_trials(trial, args=get_args())
    # print(f'Training configs: {args}')

    # init seed and device
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # data loading
    train_set, train_dataloader = data_provider(
                        root_path=args.root_path,
                        dataset_name=args.data_name,
                        flag='train',
                        seq_len=args.seq_len,
                        pred_len=args.pred_len,
                        batch_size=args.batch_size,
                        features=args.features,
                        target=args.target,
                        seed=args.seed,
                        signal_len=args.signal_len
                    )
    _, val_dataloader = data_provider(
                        root_path=args.root_path,
                        dataset_name=args.data_name,
                        flag='val',
                        seq_len=args.seq_len,
                        pred_len=args.pred_len,
                        batch_size=args.batch_size,
                        features=args.features,
                        target=args.target,
                        seed=args.seed,
                        signal_len=args.signal_len
                    )
    _, test_dataloader = data_provider(
                        root_path=args.root_path,
                        dataset_name=args.data_name,
                        flag='test',
                        seq_len=args.seq_len,
                        pred_len=args.pred_len,
                        batch_size=args.batch_size,
                        features=args.features,
                        target=args.target,
                        seed=args.seed,
                        signal_len=args.signal_len
                    )

    # init framelet model
    dframes = get_frame(args.frame_type)  # 'Haar' or 'Linear'
    approx = np.array(
            [cheb_approx(dframes[j], args.cheb_order) for j in range(len(dframes))]
        )   

    model = FreqTimeUFGV2(seq_length=args.seq_len,
                            signal_length=args.signal_len, 
                            pred_length=args.pred_len, 
                            hidden_size=args.hidden_size,
                            embed_size=args.embed_size,
                            num_ts=train_set.num_ts,
                            device=device,
                            approx=approx,
                            s=args.s,
                            lev=args.lev,
                            num_topk=args.k).to(device)

    my_optim = torch.optim.RMSprop(params=model.parameters(),
                                lr=args.learning_rate, weight_decay=args.decay_rate,
                                eps=1e-08)

    my_scheduler = ReduceLROnPlateau(my_optim, 'min', patience=10)

    forecast_loss = nn.MSELoss(reduction='mean').to(device)

    def validate(model, vali_loader):
        model.eval()
        # model.train()
        cnt = 1
        loss_total = 0
        preds = []
        trues = []
        for i, (x, y) in enumerate(vali_loader):
            cnt += 1
            y = y.float().to(device)
            x = x.float().to(device)
            forecast = model(x)
            if args.long_pred == 0:
                y = y.permute(0, 2, 1).contiguous()
            elif args.long_pred == 1:
                forecast = forecast[:,:,-1]
                y = y.permute(0, 2, 1).contiguous()[:,:,-1]
            loss = forecast_loss(forecast, y)
            loss_total += float(loss)
            forecast = forecast.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            preds.append(forecast)
            trues.append(y)
        model.train()
        return loss_total/cnt

    def test(model, test_dataloader):
        model.eval()
        preds = []
        trues = []
        for index, (x, y) in enumerate(test_dataloader):
            y = y.float().to(device)
            x = x.float().to(device)
            forecast = model(x)
            if args.long_pred == 0:
                y = y.permute(0, 2, 1).contiguous()
            elif args.long_pred == 1:
                forecast = forecast[:,:,-1]
                y = y.permute(0, 2, 1).contiguous()[:,:,-1]
            loss = forecast_loss(forecast, y)
            forecast = forecast.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            preds.append(forecast)
            trues.append(y)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        sc= evaluate(trues, preds)
        if args.long_pred == 1:
            print('-------------------------Start Test------------------------------')
            print(f'| TEST | MSE {loss:7.9f} | MAE {sc[1]:7.9f} | RMSE {sc[2]:7.9f}|')
            print('-----------------------------------------------------------------')
        else:
            print('-------------------------Start Test------------------------------')
            print(f'| TEST | MAPE {sc[0]:7.9%} | MAE {sc[1]:7.9f} | RMSE {sc[2]:7.9f}|')
            print('-----------------------------------------------------------------')
        return sc[0], sc[1], sc[2]


    best_val = 1*10^5
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_dataloader):
            cnt += 1
            y = y.float().to(device)
            x = x.float().to(device)
            forecast = model(x) 
            if args.long_pred == 0:
                y = y.permute(0, 2, 1).contiguous()
            elif args.long_pred == 1:
                forecast = forecast[:,:,-1]
                y = y.permute(0, 2, 1).contiguous()[:,:,-1]
            loss = forecast_loss(forecast, y)
            loss.backward()
            my_optim.step()
            loss_total += float(loss)

        val_loss = validate(model, val_dataloader)
        print('| Epoch{:3d} | time: {:5.2f}s | train_loss {:5.4f} | val_loss {:5.4f} |'.format(
            epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))  # noqa: E501
        my_scheduler.step(val_loss)

        if best_val >= val_loss:
            best_val = val_loss
            test_mape, test_mae, test_rmse = test(model, test_dataloader)
    print(test_mae)
