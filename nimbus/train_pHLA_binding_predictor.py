import os
import argparse
from datetime import datetime
from pprint import pformat
import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from nimbus.predictors import pHLABindingPredictor
from nimbus.globals import DEVICE, CHECKPOINT_PATH, SEED, LOGGER_LEVEL, N_WORKERS
from nimbus.utils import LoggerFactory, balance_binary_data
from nimbus.data_processing import pHLADataset

logger = LoggerFactory.get_logger(__name__, LOGGER_LEVEL)
torch.set_float32_matmul_precision('medium')  # or 'high' or 'low'


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
                        default='',
                        help='File containing validation data. Relative file path to data_dir. If empty, the validation set is splitted from the training set')
    parser.add_argument('--test_data_file',
                        dest='test_data_file',
                        type=str,
                        default='pHLA_binding/NetMHCpan_dataset/test_set_peptides_data_MaxLenPep15_hla_ABC.csv.gz',
                        help='File containing test data. Relative file path to data_dir.')
    parser.add_argument('--hla_fp_file',
                        dest='hla_fp_file',
                        type=str,
                        default='hla_fingerprints/hla_af_patch_emb_patch_r18_pt400.npy',
                        help='File containing HLA fingerprints')
    parser.add_argument('--hla_fp_data_file',
                        dest='hla_fp_data_file',
                        type=str,
                        default='hla_fingerprints/hla_af_patch_info_patch_r18_pt400.csv',
                        help='File with HLA allele names and their corresponding indices in `hla_fp_file`')
    parser.add_argument('--use_augmented_hla_data',
                        dest='use_augmented_hla_data',
                        action='store_true',
                        help='If set, use the augmented HLA data provided in the flags'
                             '`--hla_fp_file` and `--hla_fp_data_file`')
    parser.add_argument('--exp_name',
                        dest='exp_name',
                        type=str,
                        default='version',
                        help='Name of the experiment. It will be used to create the loggers')
    # Behavior parameters
    parser.add_argument('--train',
                        dest='train',
                        action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--predict',
                        dest='predict',
                        action='store_true',
                        help='Whether to predict on test set')
    parser.add_argument('--use_train_data_file_splits',
                        dest='use_train_data_file_splits',
                        type=str,
                        default='',
                        help='If not empty, it uses the splits in train_data_file'
                             'under column `split`. The expected string should '
                             'have the format split_train1,split_train2:split_val1.'
                             'Eg: if we have 5 splits numbered from 0 to 4, one '
                             'could set 0,1,2,3:4 to use splits 0,1,2,3 to train '
                             'and 4 to validate.')
    parser.add_argument('--balance_train_val',
                        dest='balance_train_val',
                        action='store_true',
                        help='Balance the training and validation data by under-sampling the majority class')
    parser.add_argument('--balance_test',
                        dest='balance_test',
                        action='store_true',
                        help='Balance the validation data by under-sampling the majority class')
    parser.add_argument('--split_train',
                        dest='split_train',
                        action='store_true',
                        help='Whether to split the training data into training and validation sets. '
                             'If no validation file is provided, the validation set is split from the training data (this flag will be set as True).')
    parser.add_argument('--split_ratio',
                        dest='split_ratio',
                        type=float,
                        default=0.8,
                        help='Ratio of the training data to be used for training')
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
    parser.add_argument('--pep_seq_len',
                        dest='pep_seq_len',
                        type=int,
                        default=15,
                        help='Max length of peptide sequence. Shorter sequences are padded')
    parser.add_argument('--hla_n_fp',
                        dest='hla_n_fp',
                        type=int,
                        default=400,
                        help='Number of Fingerprints points sampled for each HLA surface')
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
                        default='',
                        help='Path to a pretrained model')
    args = parser.parse_args()
    # Generate config dictionary
    config_dict = vars(args)
    return config_dict


def load_model(config):
    if config['pretrained_filename'] != '' and os.path.isfile(config['pretrained_filename']):
        logger.info(f"Loading pretrained model {config['pretrained_filename']}")
        model = pHLABindingPredictor.load_from_checkpoint(config['pretrained_filename'])
    else:
        model = pHLABindingPredictor(**config)
    return model


def train_predictor(model, train_loader, val_loader, config):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = config['checkpoint_path']
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
        # Log in different directories so version_X coincide between loggers
        logger=[CSVLogger(os.path.join(root_dir, 'csv_logger'),
                          name=config['exp_name']),
                TensorBoardLogger(os.path.join(root_dir, 'tensorboard_logger'),
                                  name=config['exp_name'])],
        callbacks=[EarlyStopping(patience=20, monitor='val_loss'),
                   checkpoint_callback],
        accelerator="auto",
        devices=1,
        max_epochs=config['n_epochs'],
        gradient_clip_val=0.5,
        # fast_dev_run=True,  # For testing
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    trainer.fit(model, train_loader, val_loader)

    # TODO Output the best model from training
    # model = model.to(config['device']) # Not necessary as the model is already on the device

    return model


def test_predictor(model, test_loader, config):
    root_dir = config['checkpoint_path']
    if not config['exp_name'].startswith('test'):
        config['exp_name'] = f"test_{config['exp_name']}"
    trainer = L.Trainer(
        default_root_dir=root_dir,
        logger=[TensorBoardLogger(os.path.join(root_dir, 'tensorboard_logger'),
                                  name=config['exp_name'])],
        accelerator="auto",
        devices=1,
    )
    # Try to add hparams in Tensorboard but doesn't work
    # trainer.logger.experiment.add_hparams(
    #     hparam_dict=config,
    #     metric_dict=dict())
    trainer.test(model, test_loader)


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
    config['device'] = DEVICE.type
    config['seed'] = SEED
    config['n_workers'] = N_WORKERS
    config['checkpoint_path'] = CHECKPOINT_PATH
    config['date'] = datetime.today().strftime('%Y-%m-%d')
    # TODO Check if the config_dict is correct
    logger.debug(f'Parsed Configuration:\n{pformat(config)}')

    # Get filenames
    train_file = os.path.join(config['data_dir'], config['train_data_file'])
    val_file = os.path.join(config['data_dir'], config['val_data_file'])
    test_file = os.path.join(config['data_dir'], config['test_data_file'])
    hla_fp_file = os.path.join(config['data_dir'], config['hla_fp_file'])
    hla_fp_data_file = os.path.join(config['data_dir'], config['hla_fp_data_file'])

    # Load df with HLA names as index to get the index of the HLA in the hla_fp
    if config['use_augmented_hla_data']:
        # To process raw data
        # newd = {f"{'-'.join(k[0].split('-')[:2])}{'-'.join(k[0].split('_')[0].split('-')[-2:])}": d[k] for k in d.keys()}
        tmp_data = pd.read_pickle(hla_fp_data_file)
        hla_fp_data = {k: i for i, k in enumerate(tmp_data.keys())}
    else:
        hla_fp_data = pd.read_csv(
            hla_fp_data_file,
            index_col=1,
            names=['index'],
            header=0).to_dict()['index']
        logger.debug('Loaded HLA data file successfully')
    hla_fp = np.load(hla_fp_file)
    # Dict of HLA alleles and their fingerprints
    hla_fp_dict = {hla: torch.Tensor(hla_fp[idx]) for hla, idx in hla_fp_data.items()}
    logger.debug(f'Created hla_fp_dict (which should contain HLA allele as key'
                 f'and the corresponding fingerprint as value): \n{pformat(hla_fp_dict)}')

    model = load_model(config)
    logger.info('Model loaded successfully')

    if config['train']:
        train_peptide_data = pd.read_csv(train_file)
        logger.debug('Loaded training data successfully')
        if config['use_train_data_file_splits'] != '':
            logger.info(f'Using the splits in the training data file {train_file}')
            splits = config['use_train_data_file_splits'].split(':')
            train_splits = [str(x) for x in splits[0].split(',')]
            val_splits = [str(x) for x in splits[1].split(',')]
            # Check if train and val splits are overlapping
            if len(set(train_splits).intersection(set(val_splits))) > 0:
                logger.warning('Train and validation splits are overlapping. '
                               'This may lead to data leakage. Please check the'
                               ' splits.')
            # Ensure that the split column is a string
            train_peptide_data['split'] = train_peptide_data.split.astype(str)
            val_peptide_data = train_peptide_data[train_peptide_data['split'].isin(val_splits)]
            train_peptide_data = train_peptide_data[train_peptide_data['split'].isin(train_splits)]
            # Assert not empty datasets
            assert train_peptide_data.shape[0] != 0, "There's no train data"
            assert val_peptide_data.shape[0] != 0, "There's no validation data"
            logger.info(f'After splitting, training data has shape {train_peptide_data.shape}, '
                        f'Validation data has shape {val_peptide_data.shape}')

        elif config['split_train'] or config['val_data_file'] == '':
            logger.info(f'Splitting training data into training and validation '
                        f'sets with ratio {config["split_ratio"]}')
            config['split_train'] = True  # Update the config to reflect the split
            # use sklearn train_test_split
            from sklearn.model_selection import train_test_split
            train_peptide_data, val_peptide_data = train_test_split(
                train_peptide_data,
                test_size=1-config['split_ratio'],
                random_state=config['seed'],
                stratify=train_peptide_data['label'],
                shuffle=True
            )
            logger.info(f'After splitting, training data has shape {train_peptide_data.shape}, '
                        f'Validation data has shape {val_peptide_data.shape}')
        else:
            logger.info(f'Using the provided validation data {config["val_data_file"]}')
            val_peptide_data = pd.read_csv(val_file)

        if config['balance_train_val']:
            logger.debug(f'Balancing training and validation data. Current shapes: '
                         f'{train_peptide_data.shape}, {val_peptide_data.shape}')
            logger.info('Balancing training data')
            train_peptide_data = balance_binary_data(train_peptide_data, 'label', seed=config['seed'])
            val_peptide_data = balance_binary_data(val_peptide_data, 'label', seed=config['seed'])
            logger.debug(f'Balanced training and validation data. New shapes: '
                         f'{train_peptide_data.shape}, {val_peptide_data.shape}')

        # Create datasets
        train_dataset = pHLADataset(
            peptide_seq_arr=train_peptide_data['peptide'].values,
            hla_names_arr=train_peptide_data['hla_allele'].values,
            hla_fp_dict=hla_fp_dict,
            labels=train_peptide_data['label'].values,
            has_augmented_hla=config['use_augmented_hla_data']
        )
        logger.debug(f'Sample training data shapes: \n'
                     f'peptide: {train_dataset[0][0].shape}, \n'
                     f'hla: {train_dataset[0][1].shape} \n'
                     f'label: {train_dataset[0][2].shape}')
        val_dataset = pHLADataset(
            peptide_seq_arr=val_peptide_data['peptide'].values,
            hla_names_arr=val_peptide_data['hla_allele'].values,
            hla_fp_dict=hla_fp_dict,
            labels=val_peptide_data['label'].values,
            has_augmented_hla=config['use_augmented_hla_data']
        )

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=config['n_workers'],
            shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['n_workers'],
            shuffle=False)

        model = train_predictor(
            model,
            train_loader,
            val_loader,
            config
        )
        logger.info('Training completed successfully')

    logger.info('Skipping training. If you want to train the model, use '
                'the flag `--train` to enable training.')

    # Test the model
    if config['predict']:
        if config['pretrained_filename'] == '' and config['train']:
            logger.warning('No pretrained model provided. Testing the model '
                           'after training.')
        elif config['pretrained_filename'] == '' and not config['train']:
            logger.warning('No pretrained model provided. Testing the model '
                           'without training. This may not be useful unless '
                           'you are debugging.')
        test_peptide_data = pd.read_csv(test_file)
        logger.debug('Loaded test data successfully')
        if config['balance_test']:
            logger.debug(f'Balancing test data. Current shape: {test_peptide_data.shape}')
            logger.info('Balancing test data')
            test_peptide_data = balance_binary_data(test_peptide_data, 'label', seed=config['seed'])
            logger.debug(f'Balanced validation data. New shape: {test_peptide_data.shape}')

        test_dataset = pHLADataset(
            peptide_seq_arr=test_peptide_data['peptide'].values,
            hla_names_arr=test_peptide_data['hla_allele'].values,
            hla_fp_dict=hla_fp_dict,
            labels=test_peptide_data['label'].values
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['n_workers'],
            shuffle=False
        )
        test_predictor(model, test_loader, config)
        logger.info('Testing completed successfully')
    