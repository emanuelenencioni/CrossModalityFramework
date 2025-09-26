from torch.utils.data import DataLoader
from dataset.dsec import collate_ssl

def build_from_config(train_ds, test_ds, cfg):
    """
    Build dataloaders for training and testing datasets.

    Args:
        cfg (dict): Configuration dictionary containing dataset parameters.
        train_ds (Dataset): Training dataset.
        test_ds (Dataset): Testing dataset.
        collate_fn (callable): Function to collate samples into batches.

    Returns:
        DataLoader: DataLoader for the training dataset.
        DataLoader: DataLoader for the testing dataset.
    """
    assert 'batch_size' in cfg['dataset'].keys(), " specify 'batch_size' dataset param"
    num_workers = cfg['dataset'].get('num_workers', 2)
        
    train_dl = DataLoader(train_ds, batch_size=cfg['dataset']['batch_size'], 
                          num_workers=num_workers, shuffle=True, collate_fn=collate_ssl, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=cfg['dataset']['batch_size'], 
                         num_workers=num_workers, shuffle=False, collate_fn=collate_ssl, pin_memory=True) if test_ds is not None else None


    return train_dl, test_dl