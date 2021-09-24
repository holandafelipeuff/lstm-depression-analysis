from typing import Dict
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from sklearn.metrics import precision_recall_fscore_support, plot_confusion_matrix

from torch.nn import LSTM

class MyLSTM(pl.LightningModule):

    def __init__(self):
        super().__init__()
    
        pt = "neuralmind/bert-base-portuguese-cased"

        self.lstm = nn.LSTM(input_size = 768, hidden_size = 64, num_layers = 1, batch_first = True)
        self.fc = nn.Linear(64, 2)
        
        self.model = AutoModel.from_pretrained(pt)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.pooler.named_parameters():
            param.requires_grad = True

        #self.save_hyperparameters()

    def _list_from_tensor(self, tensor):
        if tensor.numel() == 1:
            return [tensor.item()]
        return list(tensor.cpu().detach().numpy())

    def _list_from_list_of_tensor(self, list_of_tensors):
        list_to_return = []
        for item in list_of_tensors:
            list_to_return = list_to_return + self._list_from_tensor(item)
        return list_to_return

    def prepare_batch_to_lstm_model(self, batch):
        input_ids, attention_masks, labels = batch

        # Pegando a quantidade de elementos no batch
        batch_size = input_ids.size()[0]

        results_in_list = []
        for i in range(batch_size):
            result = self.model(input_ids[i], attention_masks[i])
            # Pegando apenas o pooler_out
            results_in_list.append(result[-1])            

        result_prepared_to_lstm = torch.stack(results_in_list, dim=0)

        return result_prepared_to_lstm

    '''
    def forward(self, forw_batch):
        x = self.prepare_batch_to_lstm_model(forw_batch)

        lstm_out, (ho, _) = self.lstm(x)

        x = self.fc(lstm_out[:,-1])

        print(x.squeeze())
        
        return x.squeeze()
    '''        

    def forward(self, forw_batch):
        x = self.prepare_batch_to_lstm_model(forw_batch)

        #faltou chamar aqui os 3 parâmetros sequência, h0, c0
        lstm_out, (ho, _) = self.lstm(x) 

        x = ho[-1]
        
        # x é o logit que será 
        x = self.fc(x)

        return x.squeeze()
    
    def training_step(self, train_batch, batch_idx):
        *_, labels = train_batch
        
        logits = self(train_batch)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO: verificar se é assim memso, com esse .view()
            # loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            loss = loss_fct(logits, labels)

        with torch.no_grad():
            preds = F.softmax(logits, dim=1).argmax(dim=1)
            acc = ((preds == labels).sum().float()) / len(labels)

        print({"train_loss": loss, "train_acc": acc, "labels": labels, "preds": preds})
        print("\n")

        obj_to_return = {"loss": loss, "acc": acc, "labels": labels, "preds": preds}

        return obj_to_return

    def training_epoch_end(self, outs):
        
        loss = torch.stack([x['loss'] for x in outs]).mean()
        acc = torch.stack([x['acc'] for x in outs]).mean()

        labels = []
        preds = []

        for out in outs:
            for i in range(len(out["labels"])):
                labels.append(out["labels"][i])
                preds.append(out["preds"][i])

        #labels_cpu = labels.to('cpu').numpy()
        #preds_cpu = preds.to('cpu').numpy()
        labels_cpu = self._list_from_list_of_tensor(labels)
        preds_cpu = self._list_from_list_of_tensor(preds)

        #precision, recall, fscore, _ = precision_recall_fscore_support(labels_cpu, preds_cpu, average="binary")
        precision, recall, fscore, _ = precision_recall_fscore_support(labels_cpu, preds_cpu, average="binary")
        
        obj_to_return = {
            "train_epoch_loss": loss.item(), 
            "train_epoch_acc": acc.item(), 
            "train_epoch_precision": precision, 
            "train_epoch_recall": recall, 
            "train_epoch_fscore": fscore
        }
        
        print("\n\nTRAIN EPOCH RESULTS")
        print(obj_to_return)
        print("\n\n")

        #plot_confusion_matrix(
        #    labels, preds, ["Not Depressed", "Depressed"]
        #)

        #return obj_to_return

    def validation_step(self, val_batch, batch_idx):
        *_, labels = val_batch
        
        logits = self(val_batch)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO: verificar se é assim memso, com esse .view()
            # loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            loss = loss_fct(logits, labels)
        
        preds = F.softmax(logits, dim=1).argmax(dim=1)
        acc = ((preds == labels).sum().float()) / len(labels)

        print({"val_loss": loss, "val_acc": acc, "labels": labels, "preds": preds})
        print("\n")

        obj_to_return = {"loss": loss, "acc": acc, "labels": labels, "preds": preds}
        
        return obj_to_return


    # NESSE CARA QUE A GNT VAI FAZER O SELF.LOG_DICT
    # def validation_epoch_end(self, outs):

    def validation_epoch_end(self, outs):
        
        loss = torch.stack([x['loss'] for x in outs]).mean()
        acc = torch.stack([x['acc'] for x in outs]).mean()

        labels = []
        preds = []

        for out in outs:
            for i in range(len(out["labels"])):
                labels.append(out["labels"][i])
                preds.append(out["preds"][i])

        #labels_cpu = labels.to('cpu').numpy()
        #preds_cpu = preds.to('cpu').numpy()
        labels_cpu = self._list_from_list_of_tensor(labels)
        preds_cpu = self._list_from_list_of_tensor(preds)

        #precision, recall, fscore, _ = precision_recall_fscore_support(labels_cpu, preds_cpu, average="binary")
        precision, recall, fscore, _ = precision_recall_fscore_support(labels_cpu, preds_cpu, average="binary")

        obj_to_return = {
            "val_epoch_loss": loss.item(), 
            "val_epoch_acc": acc.item(), 
            "val_epoch_precision": precision, 
            "val_epoch_recall": recall, 
            "val_epoch_fscore": fscore
        }

        print("\n\nVAL EPOCH RESULTS")
        print(obj_to_return)
        print("\n\n")

        #self.log_dict(obj_to_return)
        self.log('val_epoch_acc', obj_to_return["val_epoch_acc"])

        #plot_confusion_matrix(
        #    labels, preds, ["Not Depressed", "Depressed"]
        #)

    # TODO: Verificar se isso está correto
    def test_step(self, test_batch, batch_idx):
        *_, labels = test_batch
        
        logits = self(test_batch)

        # TODO: verificar se faz sentido calcular loss no teste
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO: verificar se é assim memso, com esse .view()
            # loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            loss = loss_fct(logits, labels)
        
        preds = F.softmax(logits, dim=1).argmax(dim=1)
        acc = ((preds == labels).sum().float()) / len(labels)

        print({"test_loss": loss, "test_acc": acc, "labels": labels, "preds": preds})
        print("\n")

        obj_to_return = {"loss": loss, "acc": acc, "labels": labels, "preds": preds}
        
        return obj_to_return

    
    def test_epoch_end(self, outs):
        
        loss = torch.stack([x['loss'] for x in outs]).mean()
        acc = torch.stack([x['acc'] for x in outs]).mean()

        labels = []
        preds = []

        for out in outs:
            for i in range(len(out["labels"])):
                labels.append(out["labels"][i])
                preds.append(out["preds"][i])

        #labels_cpu = labels.to('cpu').numpy()
        #preds_cpu = preds.to('cpu').numpy()
        labels_cpu = self._list_from_list_of_tensor(labels)
        preds_cpu = self._list_from_list_of_tensor(preds)

        #precision, recall, fscore, _ = precision_recall_fscore_support(labels_cpu, preds_cpu, average="binary")
        precision, recall, fscore, _ = precision_recall_fscore_support(labels_cpu, preds_cpu, average="binary")
        
        obj_to_return = {
            "test_epoch_loss": loss.item(), 
            "test_epoch_acc": acc.item(), 
            "test_epoch_precision": precision, 
            "test_epoch_recall": recall, 
            "test_epoch_fscore": fscore
        }

        print("\n\nTEST EPOCH RESULTS")
        print(obj_to_return)
        print("\n\n")

        #plot_confusion_matrix(
        #    labels, preds, ["Not Depressed", "Depressed"]
        #)

        self.log('results', obj_to_return)
        return obj_to_return


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)