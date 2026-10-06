from torch.utils.data import DataLoader

from src.data_loader import ForecastDataset


def data_provider(args, flag):
    dataset = ForecastDataset(args.root_path, args.data_name, flag,
                              args.seq_len, args.pred_len)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(flag != "test"), drop_last=False)
    return dataset, loader
