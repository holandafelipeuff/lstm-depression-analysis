import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import click

from lstm_depression_analysis.data.dataset import DepressionCorpusTwitter, DepressionCorpusInsta

from lstm_depression_analysis.models.models import MyLSTM
from lstm_depression_analysis.training.training import train_my_lstm

def free_memory(model, trainer, lstm_dataset_test, data_loader_test, lstm_dataset_train, data_loader_train, lstm_dataset_val, data_loader_val):
        del model
        del trainer
        del lstm_dataset_test
        del data_loader_test
        del lstm_dataset_train
        del data_loader_train
        del lstm_dataset_val
        del data_loader_val

        torch.cuda.empty_cache()

def collate_fn(batch):

    # O retorno é 
    
    # (tamanho do batch, 
    #  LIMITADO EM 64 - tamanho da sequencia (número de posts do usuário que tem mais) (podemos limitar num número máx de posts), 
    #  512)

    post_length_limit = 64

    results_input_ids = []
    results_attention_mask = []
    results_labels = []
    
    for user_data in batch:
        
        tokenizer_info = user_data[0]
        label = user_data[1]
        
        results_input_ids.append(tokenizer_info["input_ids"][:post_length_limit])
        results_attention_mask.append(tokenizer_info["attention_mask"][:post_length_limit])
        #results_input_ids.append(tokenizer_info["input_ids"])
        #results_attention_mask.append(tokenizer_info["attention_mask"])

        results_labels.append(label)

    results_labels = torch.LongTensor(results_labels)

    results_input_ids.sort(key=len, reverse=True)
    results_attention_mask.sort(key=len, reverse=True)

    padded_results_input_ids = pad_sequence(sequences = results_input_ids, batch_first = True, padding_value = 0.0)
    padded_results_attention_mask = pad_sequence(sequences = results_attention_mask, batch_first = True, padding_value = 0.0)

    return (padded_results_input_ids, padded_results_attention_mask, results_labels)

def calculate_results(results_mean_temp):

    results_mean_to_return = {key:{} for key in results_mean_temp}

    for key in results_mean_temp:
        test_epoch_acc = 0
        test_epoch_fscore = 0
        test_epoch_loss = 0
        test_epoch_precision = 0
        test_epoch_recall = 0

        for i in range(len(results_mean_temp[key])):
            test_epoch_acc += results_mean_temp[key][i]['results']['test_epoch_acc']
            test_epoch_fscore += results_mean_temp[key][i]['results']['test_epoch_fscore']
            test_epoch_loss += results_mean_temp[key][i]['results']['test_epoch_loss']
            test_epoch_precision += results_mean_temp[key][i]['results']['test_epoch_precision']
            test_epoch_recall += results_mean_temp[key][i]['results']['test_epoch_recall']

        test_epoch_acc = test_epoch_acc / len(results_mean_temp[key])
        test_epoch_fscore = test_epoch_fscore / len(results_mean_temp[key])
        test_epoch_loss = test_epoch_loss / len(results_mean_temp[key])
        test_epoch_precision = test_epoch_precision / len(results_mean_temp[key])
        test_epoch_recall = test_epoch_recall / len(results_mean_temp[key])

        results_mean_to_return[key] = {
            "test_epoch_acc": test_epoch_acc,
            "test_epoch_fscore": test_epoch_fscore,
            "test_epoch_loss": test_epoch_loss,
            "test_epoch_precision": test_epoch_precision,
            "test_epoch_recall": test_epoch_recall,
        }

    return results_mean_to_return

## Experimentos Não-Cross

def run_real_train_val_twitter_test_twitter_experiment(
        batch_size,
        epochs,
        shuffle,
        gpu = None,
        periods=[60, 212, 365],
        datasets=10):

    datasets = list(range(0,datasets))
    
    results_mean_temp = {d:[] for d in periods}
        
    for days in periods:
        for dataset in datasets:
            
            print(f"Training model for {days} days and dataset {dataset}...")

            # Pegando dataloader de teste
            lstm_dataset_test = DepressionCorpusTwitter(days, dataset, "test", batch_size)
            data_loader_test = torch.utils.data.DataLoader(dataset = lstm_dataset_test, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de treino
            lstm_dataset_train = DepressionCorpusTwitter(days, dataset, "train", batch_size)
            data_loader_train = torch.utils.data.DataLoader(dataset = lstm_dataset_train, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de val
            lstm_dataset_val = DepressionCorpusTwitter(days, dataset, "val", batch_size)
            data_loader_val = torch.utils.data.DataLoader(dataset = lstm_dataset_val, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            model = MyLSTM()

            trainer = train_my_lstm(model = model, gpu = gpu, data_loader_train = data_loader_train, data_loader_val = data_loader_val, batch_size = batch_size, epochs = epochs, collate_fn = collate_fn, shuffle = shuffle)

            print(f"Testing model for {days} days and dataset {dataset}...")

            results = trainer.test(test_dataloaders = data_loader_test)

            for result in results:
                results_mean_temp[days].append(result)


            free_memory(model, trainer, lstm_dataset_test, data_loader_test, lstm_dataset_train, data_loader_train, lstm_dataset_val, data_loader_val)

    results_mean = calculate_results(results_mean_temp)

    print(results_mean_temp)
    print(results_mean)
    
    with open('results-twitter-twitter.txt', 'w') as f:
        f.write(str(results_mean_temp))
        f.write(str("\n\n"))
        f.write(str(results_mean))

def run_real_train_val_insta_test_insta_experiment(
        batch_size,
        epochs,
        shuffle,
        gpu = None,
        periods=[60, 212, 365],
        datasets=10):

    datasets = list(range(0,datasets))

    results_mean_temp = {d:[] for d in periods}
        
    for days in periods:
        for dataset in datasets:
            
            print(f"Training model for {days} days and dataset {dataset}...")

            # Pegando dataloader de teste
            lstm_dataset_test = DepressionCorpusInsta(days, dataset, "test", batch_size)
            data_loader_test = torch.utils.data.DataLoader(dataset = lstm_dataset_test, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de treino
            lstm_dataset_train = DepressionCorpusInsta(days, dataset, "train", batch_size)
            data_loader_train = torch.utils.data.DataLoader(dataset = lstm_dataset_train, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de val
            lstm_dataset_val = DepressionCorpusInsta(days, dataset, "val", batch_size)
            data_loader_val = torch.utils.data.DataLoader(dataset = lstm_dataset_val, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            #model = MyLSTM(scheduler_args, optimizer_args)
            model = MyLSTM()

            trainer = train_my_lstm(model = model, gpu = gpu, data_loader_train = data_loader_train, data_loader_val = data_loader_val, batch_size = batch_size, epochs = epochs, collate_fn = collate_fn, shuffle = shuffle)

            print(f"Testing model for {days} days and dataset {dataset}...")

            results = trainer.test(test_dataloaders = data_loader_test)

            for result in results:
                results_mean_temp[days].append(result)

            free_memory(model, trainer, lstm_dataset_test, data_loader_test, lstm_dataset_train, data_loader_train, lstm_dataset_val, data_loader_val)

    results_mean = calculate_results(results_mean_temp)

    print(results_mean_temp)
    print(results_mean)

    with open('results-insta-insta.txt', 'w') as f:
        f.write(str(results_mean_temp))
        f.write(str("\n\n"))
        f.write(str(results_mean))

## Experimentos Cross

def run_real_train_val_insta_test_twitter_experiment(
        batch_size,
        epochs,
        shuffle,
        gpu = None,
        periods=[60, 212, 365],
        datasets=10):

    datasets = list(range(0,datasets))

    results_mean_temp = {d:[] for d in periods}
        
    for days in periods:
        for dataset in datasets:
            
            print(f"Training model for {days} days and dataset {dataset}...")

            # Pegando dataloader de teste
            lstm_dataset_test = DepressionCorpusTwitter(days, dataset, "test", batch_size)
            data_loader_test = torch.utils.data.DataLoader(dataset = lstm_dataset_test, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de treino
            lstm_dataset_train = DepressionCorpusInsta(days, dataset, "train", batch_size)
            data_loader_train = torch.utils.data.DataLoader(dataset = lstm_dataset_train, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de val
            lstm_dataset_val = DepressionCorpusInsta(days, dataset, "val", batch_size)
            data_loader_val = torch.utils.data.DataLoader(dataset = lstm_dataset_val, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            model = MyLSTM()

            trainer = train_my_lstm(model = model, gpu = gpu, data_loader_train = data_loader_train, data_loader_val = data_loader_val, batch_size = batch_size, epochs = epochs, collate_fn = collate_fn, shuffle = shuffle)

            print(f"Testing model for {days} days and dataset {dataset}...")

            results = trainer.test(test_dataloaders = data_loader_test)

            for result in results:
                results_mean_temp[days].append(result)

            free_memory(model, trainer, lstm_dataset_test, data_loader_test, lstm_dataset_train, data_loader_train, lstm_dataset_val, data_loader_val)

    results_mean = calculate_results(results_mean_temp)

    print(results_mean_temp)
    print(results_mean)

    with open('results-insta-twitter.txt', 'w') as f:
        f.write(str(results_mean_temp))
        f.write(str("\n\n"))
        f.write(str(results_mean))

def run_real_train_val_twitter_test_insta_experiment(
        batch_size,
        epochs,
        shuffle,
        gpu = None,
        periods=[60, 212, 365],
        datasets=10):

    datasets = list(range(0,datasets)) 

    results_mean_temp = {d:[] for d in periods}
        
    for days in periods:
        for dataset in datasets:
            
            print(f"Training model for {days} days and dataset {dataset}...")

            # Pegando dataloader de teste
            lstm_dataset_test = DepressionCorpusInsta(days, dataset, "test", batch_size)
            data_loader_test = torch.utils.data.DataLoader(dataset = lstm_dataset_test, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de treino
            lstm_dataset_train = DepressionCorpusTwitter(days, dataset, "train", batch_size)
            data_loader_train = torch.utils.data.DataLoader(dataset = lstm_dataset_train, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            # Pegando dataloader de val
            lstm_dataset_val = DepressionCorpusTwitter(days, dataset, "val", batch_size)
            data_loader_val = torch.utils.data.DataLoader(dataset = lstm_dataset_val, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)

            model = MyLSTM()

            trainer = train_my_lstm(model = model, gpu = gpu, data_loader_train = data_loader_train, data_loader_val = data_loader_val, batch_size = batch_size, epochs = epochs, collate_fn = collate_fn, shuffle = shuffle)

            print(f"Testing model for {days} days and dataset {dataset}...")

            results = trainer.test(test_dataloaders = data_loader_test)

            for result in results:
                results_mean_temp[days].append(result)

            free_memory(model, trainer, lstm_dataset_test, data_loader_test, lstm_dataset_train, data_loader_train, lstm_dataset_val, data_loader_val)

    results_mean = calculate_results(results_mean_temp)

    print(results_mean_temp)
    print(results_mean)
    
    with open('results-twitter-insta.txt', 'w') as f:
        f.write(str(results_mean_temp))
        f.write(str("\n\n"))
        f.write(str(results_mean))

@click.command()
@click.option(
    "--batch_size", required=True, help=f"Size of the batches used in the experiment", type=click.INT
)
@click.option(
    "--epochs", required=True, help=f"Number of the epochs used in the experiment", type=click.INT
)
@click.option(
    "--shuffle", required=True, help=f"If should shuffle or not 1 - True, 0 - False", type=click.INT
)
@click.option(
    "--gpu", required=True, help=f"Number of the GPU used in the experiment", type=click.INT
)
@click.option(
    "--periods", required=True, help=f"Which periods should be used 0 - [60], 1 - [60, 212], 2 - [60, 212, 365]", type=click.INT
)
@click.option(
    "--datasets", required=True, help=f"Number of datasets used in the experiment", type=click.INT
)
@click.option(
    "--experiment", required=True, help=f"Which experiment should be run 0 - Train and Test on Insta, 1 - Train and Test on Twitter, 2 - Train on Insta and Test on Twitter, 3 - Train on Twitter and Test on Insta, ", type=click.INT
)
def run_experiment(
    batch_size: int,
    epochs: int,
    shuffle: int,
    gpu: int,
    periods: int,
    datasets: int,
    experiment: int,
):
    available_shuffles = [0, 1]
    if shuffle not in available_shuffles:
        raise ValueError(
            f"Shuffle {shuffle} is not valid. Please, use one of the following: {available_shuffles}"
        )

    if shuffle == 1:
        shuffle = True
    else:
        shuffle = False

    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7, 999]
    if gpu not in available_gpus:
        raise ValueError(
            f"GPU {gpu} is not valid. Please, use one of the following: {available_gpus}"
        )

    if gpu == 999:
        gpu = None

    available_periods = [0, 1, 2]
    if periods not in available_periods:
        raise ValueError(
            f"Period {periods} is not valid. Please, use one of the following: {available_periods}"
        )

    if periods == 0:
        periods = [60]
    elif periods == 1:
        periods = [60, 212]
    else:
        periods = [60, 212, 365]

    available_datasets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if datasets not in available_datasets:
        raise ValueError(
            f"Datasets {datasets} is not valid. Please, use one of the following: {available_datasets}"
        )
    
    available_experiment = [0, 1, 2, 3]
    if experiment not in available_experiment:
        raise ValueError(
            f"Experiments {experiment} is not valid. Please, use one of the following: {available_experiment}"
        )

    if experiment == 0:
        run_real_train_val_insta_test_insta_experiment(
            batch_size,
            epochs,
            shuffle,
            gpu,
            periods,
            datasets)
    elif experiment == 1:
        run_real_train_val_twitter_test_twitter_experiment(
            batch_size,
            epochs,
            shuffle,
            gpu,
            periods,
            datasets)
    elif experiment == 2:
        run_real_train_val_insta_test_twitter_experiment(
            batch_size,
            epochs,
            shuffle,
            gpu,
            periods,
            datasets)
    else:
        run_real_train_val_twitter_test_insta_experiment(
            batch_size,
            epochs,
            shuffle,
            gpu,
            periods,
            datasets)

if __name__ == "__main__":
    run_experiment()