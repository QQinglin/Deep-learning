import numpy as np
import pandas as pd
import torch as t
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

import model
from data import ChallengeDataset
from trainer import Trainer

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='train.log',
    filemode='w',
    format='%(asctime)s - %(message)s'
)

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
df = pd.read_csv('data.csv', sep=';')
x = df['filename']
y = df[['crack', 'inactive']].values
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# create an instance of our ResNet model
def train(configs):
    # import Net
    resnet = model.ResNet()

    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    train_dataset = ChallengeDataset(x_train.values, y_train)
    val_dataset = ChallengeDataset(x_val.values, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    criterion = nn.BCELoss()

    # set up the optimizer (see t.optim)
    optimizer = t.optim.Adam(resnet.parameters(), weight_decay=configs['weight_decay'], lr=configs['lr'])

    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(
        model=resnet,
        crit=criterion,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=val_loader,
        cuda=t.cuda.is_available(),
        early_stopping_patience=5
    )

    logging.info(f"\n\n{configs} starts:\n")
    # go, go, go... call fit on trainer
    res = trainer.fit(epochs=50)

    # plot the results
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('losses.png')

weight_decay_domain = [1e-2]
lr_domain = [1e-4]

idx = 7

for lr in lr_domain:
    for weight_decay in weight_decay_domain:
        for i in range(5):
            config = {
                'weight_decay': weight_decay,
                'lr': lr,
                'iteartion': i
            }
            train(config)