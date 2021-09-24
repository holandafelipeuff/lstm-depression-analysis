from lstm_depression_analysis.settings import PATH_TO_BEST_MODEL
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lstm_depression_analysis.models.models import MyLSTM

def train_my_lstm(
        model,
        gpu,
        data_loader_train,
        data_loader_val,
        batch_size,
        epochs,
        collate_fn,
        shuffle):

    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch_acc",
        mode="max"
    )

    gradient_accumulation_steps = 1
    t_total = (len(data_loader_train) // gradient_accumulation_steps) * epochs

    gpus = [gpu]
    if gpu is not None:
        gpus = [gpu]
    else:
        gpus = None

    # Debug execution
    #trainer = pl.Trainer(fast_dev_run = True, max_steps=t_total, gpus = gpus)

    # Normal executiion
    trainer = pl.Trainer(max_steps=t_total, gpus = gpus, callbacks=[checkpoint_callback])

    '''

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, data_loader_train, data_loader_val, min_lr=1e-15, max_lr=2, num_training=800)

    # Results can be found in
    # lr_finder.results

    # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    #
    # 0.025118864315095822 <---- resultado
    # 0.3548133892335751 <------ resultado mais novo
    # 0.002476476986380026 <---- mais novo
    # 0.22116460340604702 <----- novo mas nem tão bom

    # print("\n\n\n\n\n")
    print(new_lr)
    # print("aaaaaaaaaa")
    
    # update hparams of the model
    model.hparams.lr = new_lr

    with open('new_lr.txt', 'w') as f:
        f.write(str(new_lr))

        '''

    # Fit model
    trainer.fit(model, data_loader_train, data_loader_val)

    return trainer
            

    
