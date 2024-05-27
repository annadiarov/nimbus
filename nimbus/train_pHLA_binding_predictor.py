import os
import argparse
from pprint import pformat
import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from nimbus.predictors import pHLABindingPredictor
from nimbus.globals import DEVICE, CHECKPOINT_PATH, SEED, LOGGER_LEVEL, N_WORKERS
from nimbus.utils import LoggerFactory, balance_binary_data
from nimbus.data_processing import pHLADataset

logger = LoggerFactory.get_logger(__name__, LOGGER_LEVEL)


def parse_args():
    parser = argparse.ArgumentParser(description='Train pHLA binding predictor')
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        type=str,
                        default='../data/processed',
                        help='Directory containing the data')
    parser.add_argument('--train_data_file',
                        dest='train_data_file',
                        type=str,
                        default='pHLA_binding/NetMHCpan_dataset/train_binding_affinity_peptides_data_MaxLenPep15_hla_ABC.csv',
                        help='File containing peptide, HLA allele and labels. Relative file path to data_dir.')
    parser.add_argument('--val_data_file',
                        dest='val_data_file',
                        type=str,
                        default='pHLA_binding/NetMHCpan_dataset/test_set_peptides_data_MaxLenPep15_hla_ABC.csv.gz',
                        help='File containing validation data. Relative file path to data_dir.')
    parser.add_argument('--hla_fp_file',
                        dest='hla_fp_file',
                        type=str,
                        default='hla_fingerprints/hla_af_patch_emb_netMHCpan_radius_18_pt_400.npy',
                        help='File containing HLA fingerprints')
    parser.add_argument('--hla_fp_data_file',
                        dest='hla_fp_data_file',
                        type=str,
                        default='hla_fingerprints/hla_index_netMHCpan_pseudoseq_res_representation.csv',
                        help='File with HLA allele names and their corresponding indices in `hla_fp_file`')
    # Behavior parameters
    parser.add_argument('--balance_train',
                        dest='balance_train',
                        type=bool,
                        default=True,
                        help='Balance the training data by under-sampling the majority class')
    parser.add_argument('--balance_val',
                        dest='balance_val',
                        type=bool,
                        default=True,
                        help='Balance the validation data by under-sampling the majority class')
    # Model parameters
    parser.add_argument('--pep_embedding_dim',
                        dest='pep_embedding_dim',
                        type=int,
                        default=32,
                        help='Dimension of peptide embedding')
    parser.add_argument('--hla_fp_dim',
                        dest='hla_fp_dim',
                        type=int, default=80,
                        help='Dimension of HLA fingerprint')
    parser.add_argument('--n_self_attns',
                        dest='n_self_attns',
                        type=int,
                        default=2,
                        help='Number of self-attention layers for peptide and HLA')
    parser.add_argument('--n_joint_cross_attns',
                        type=int,
                        default=4,
                        help='Number of joint cross-attention layers')
    parser.add_argument('--filip_num_heads',
                        dest='filip_num_heads',
                        type=int,
                        default=64,
                        help='Number of heads for the FILIP layer')
    parser.add_argument('--filip_dim_head',
                        dest='filip_dim_head',
                        type=int,
                        default=16,
                        help='Dimension of each head for the FILIP layer')
    parser.add_argument('--dropout',
                        dest='dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate for self-attention, cross-attention, and FILIP layers')
    parser.add_argument('--lr',
                        dest='lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate')
    parser.add_argument('--warmup',
                        dest='warmup',
                        type=int,
                        default=20,
                        help='Number of lr-warmup iterations')
    parser.add_argument('--n_iterations_cosine_cycle',
                        dest='n_iterations_cosine_cycle',
                        type=int,
                        default=100,
                        help='Number of optimization iterations in a cosine cycle (not epoch!)')
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--n_epochs',
                        dest='n_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs. If -1, train until early stopping')
    parser.add_argument('--pretrained_filename',
                        dest='pretrained_filename',
                        type=str,
                        default=None,
                        help='Path to a pretrained model')
    parser.add_argument('--mode',
                        dest='mode',
                        type=str,
                        choices=['train', 'predict'],
                        default='train',
                        help='Whether to train the model or predict on a dataset')
    args = parser.parse_args()
    # Generate config dictionary
    config_dict = vars(args)
    return config_dict


def train_predictor(train_loader, val_loader, config):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = CHECKPOINT_PATH
    os.makedirs(root_dir, exist_ok=True)
    # save epoch and val_loss in name, but specify the formatting yourself (e.g. to avoid problems with Tensorboard
    # or Neptune, due to the presence of characters like '=' or '/')
    # saves a file like: my/path/sample-mnist-epoch02-val_loss0.32.ckpt
    checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                          mode='max',
                                          filename='epoch{epoch:02d}-val_loss{val_loss:.2f}-val_acc{val_acc:.2f}',
                                          save_top_k=1,
                                          save_last=True,
                                          auto_insert_metric_name=False)

    trainer = L.Trainer(
        default_root_dir=root_dir,
        callbacks=[EarlyStopping(patience=20, monitor='train_loss'), checkpoint_callback],
        accelerator="auto",
        devices=1,
        max_epochs=-1,
        gradient_clip_val=0.5,
        # fast_dev_run=True,  # For testing
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    if config['pretrained_filename'] is not None and os.path.isfile(config['pretrained_filename']):
        print("Found pretrained model, loading...")
        model = pHLABindingPredictor.load_from_checkpoint(config['pretrained_filename'])
    else:
        model = pHLABindingPredictor(**config)

    if config['mode'] == 'train':
        trainer.fit(model, train_loader, val_loader)
    else:
        #trainer.test(model, val_loader)
        raise NotImplementedError('Prediction mode is not implemented yet')

    model = model.to(DEVICE)
    return model


if __name__ == '__main__':
    # Setting the seed
    L.seed_everything(SEED)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'Using device: {DEVICE}')

    # Parse arguments
    config = parse_args()
    # Add some globals to the config # TODO Check if this is necessary
    config['device'] = DEVICE
    config['seed'] = SEED
    config['n_workers'] = N_WORKERS
    # TODO Check if the config_dict is correct
    logger.debug(f'Parsed Configuration:\n{pformat(config)}')

    # Load data
    train_file = os.path.join(config['data_dir'], config['train_data_file'])
    val_file = os.path.join(config['data_dir'], config['val_data_file'])
    hla_fp_file = os.path.join(config['data_dir'], config['hla_fp_file'])
    hla_fp_data_file = os.path.join(config['data_dir'], config['hla_fp_data_file'])
    train_peptide_data = pd.read_csv(train_file)
    logger.debug('Loaded training data successfully')
    val_peptide_data = pd.read_csv(val_file)
    logger.debug('Loaded validation data successfully')
    # Load df with HLA names as index to get the index of the HLA in the hla_fp
    hla_fp_data = pd.read_csv(hla_fp_data_file,
                              index_col=1,
                              names=['index'],
                              header=0).to_dict()['index']
    logger.debug('Loaded HLA data file successfully')
    hla_fp = np.load(hla_fp_file)
    # Dict of HLA alleles and their fingerprints
    hla_fp_dict = {hla: torch.Tensor(hla_fp[idx]) for hla, idx in hla_fp_data.items()}
    logger.debug(f'Created hla_fp_dict (which should contain HLA allele as key'
                 f'and the corresponding fingerprint as value): \n{pformat(hla_fp_dict)}')
    if config['balance_train']:
        logger.debug(f'Balancing training data. Current shape: {train_peptide_data.shape}')
        logger.info('Balancing training data')
        train_peptide_data = balance_binary_data(train_peptide_data, 'label', seed=SEED)
        logger.debug(f'Balanced training data. New shape: {train_peptide_data.shape}')
    if config['balance_val']:
        logger.debug(f'Balancing validation data. Current shape: {val_peptide_data.shape}')
        logger.info('Balancing validation data')
        val_peptide_data = balance_binary_data(val_peptide_data, 'label', seed=SEED)
        logger.debug(f'Balanced validation data. New shape: {val_peptide_data.shape}')

    # Create datasets
    train_dataset = pHLADataset(peptide_seq_arr=train_peptide_data['peptide'].values,
                                hla_names_arr=train_peptide_data['hla_allele'].values,
                                hla_fp_dict=hla_fp_dict,
                                labels=train_peptide_data['label'].values)
    logger.debug(f'Sample training data shapes: \n'
                 f'peptide: {train_dataset[0][0].shape}, \n'
                 f'hla: {train_dataset[0][1].shape} \n'
                 f'label: {train_dataset[0][2].shape}')
    val_dataset = pHLADataset(peptide_seq_arr=val_peptide_data['peptide'].values,
                              hla_names_arr=val_peptide_data['hla_allele'].values,
                              hla_fp_dict=hla_fp_dict,
                              labels=val_peptide_data['label'].values)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=N_WORKERS,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=N_WORKERS,
        shuffle=False)

    model = train_predictor(train_loader,
                            val_loader,
                            config)
