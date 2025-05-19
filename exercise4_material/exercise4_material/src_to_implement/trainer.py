import logging
import random
import string

import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()

        # -propagate through the network
        y_hat = self._model(x)

        # Convert labels to match the output format for BCELoss
        # Labels are 0 or 1; convert to [[0, 1], [1, 0], ...] for [1, 0, ...]
        y = y.float().unsqueeze(1)  # Shape: (batch_size, n, 1) [[0], [1], ...]
        y_target = t.cat((1 - y, y), dim=1)  # Shape: (batch_size, n, 2) [[1-0,0], [1-1,1], ...]

        # -calculate the loss
        loss = self._crit(y_hat, y_target)

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        self._optim.step()

        # -return the loss
        y_hat = t.where(y_hat >= 0.5, 1, 0)
        return loss.item(), y_hat

    def val_test_step(self, x, y):
        y_hat = self._model(x)

        # Convert labels to match the output format for BCELoss
        # Labels are 0 or 1; convert to [[0, 1], [1, 0], ...] for [1, 0, ...]
        y = y.float().unsqueeze(1)  # Shape: (batch_size, 1)
        y_target = t.cat((1 - y, y), dim=1)  # Shape: (batch_size, 2)

        # propagate through the network and calculate the loss and predictions
        loss = self._crit(y_hat, y_target)

        # return the loss and the predictions
        y_hat = t.where(y_hat >= 0.5, 1, 0)
        return loss.item(), y_hat

    def train_epoch(self):
        # set training mode
        self._model.train()

        losses = [] # loss of each batch
        y_hats = [] # prediction result of each batch
        ys = [] # truth ground label

        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        for x, y in self._train_dl: # loop batch in Dataloader
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            # perform a training step for each batch
            loss, y_hat =self.train_step(x,y)
            losses.append(loss)

            # Convert predictions to binary (0 or 1) for F1 score
            preds = y_hat.cpu().detach().numpy() # Probablility of class 1
            labels = y.cpu().detach.numpy() # Shape:(batch_size,)
            y_hats.extend(preds)
            ys.extend(labels)

        # calculate the average loss for the epoch and return it
        avg_loss = sum(losses) / len(losses) # each epoch
        f1 = f1_score(ys, y_hats, average='weighted')
        return  avg_loss,f1

    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()

        losses = [] # loss of each batch
        y_hats = [] # prediction result of each batch
        ys = [] # truth ground label

        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            # iterate through the validation set
            for x, y in self._val_test_dl: # loop batch in Dataloader
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                # perform a validation step
                loss, y_hat = self.val_test_step(x,y)
                # save the predictions and the labels for each batch
                losses.append(loss)

                # Convert predictions to binary (0 or 1) for F1 score
                preds = y_hat.cpu().detach().numpy()  # Probablility of class 1
                labels = y.cpu().detach.numpy()  # Shape:(batch_size,)
                y_hats.extend(preds)
                ys.extend(labels)

        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        avg_loss = sum(losses) / len(losses)
        # Calculate F1 score
        f1 =f1_score(ys, y_hats, average='weighted')
        print(f"Validation Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")

        # return the loss and print the calculated metrics
        return  avg_loss, f1
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        metrics_training = [] # f1 score
        metrics = []

        # Early stopping variables
        best_val_loss = float('inf') # for minimum validation loss
        best_f1 = 0
        patience_count = 0 # early stopping
        epoch = 0

        # Random prefix for checkpoint files
        prefix = "".join(random.choice(string.ascii_lowercase) for _ in range(6))
        logging.info(f"Epoch file prefix: {prefix}\n")

        while True:
            # stop by epoch number
            if epoch > 0 and epoch >= epochs:
                break

            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss, f1_training = self.train_epoch() # results of training
            val_loss, f1 = self.val_test() # results of test

            # append the losses to the respective lists
            train_losses.append(train_loss)
            metrics_training.append(f1_training)

            val_losses.append(val_loss)
            metrics.append(f1)

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if val_loss < best_val_loss or f1 > best_f1:
                best_val_loss = min(val_loss,best_val_loss)
                best_f1 = max(f1, best_f1)
                patience_count = 0 # Performance improves
                self.save_checkpoint(f"{prefix}_{epoch}")
                logging.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"F1 Training: {f1_training:.4f}, F1 Score: {f1:.4f}, Model saved as {prefix}_{epoch}"
                )
            else:
                patience_count += 1 # performance no improvement
                logging.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"F1 Training: {f1_training:.4f}, F1 Score: {f1:.4f}"
                )

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self._early_stopping_patience > 0 and patience_count >= self._early_stopping_patience: # in last epochs performance no improvement
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            epoch += 1

            # return the losses for both training and validation
            return train_losses, val_losses, metrics_training, metrics
