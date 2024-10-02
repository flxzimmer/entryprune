# (based on dev version 71)

import torch
import torch.nn as nn
import numpy as np


class BaseRCN:
    def __init__(self, relevant, epochs=200, patience=None, lr=1e-3, n_hidden=10, shrink=False, batch_size=None, patience_ident=None,switch_interval = 1,n_updates = None,final_netsize = None, method = "gradients",candidate_perc = None):

        self.relevant = relevant
        self.epochs = int(epochs)
        self.lr = lr
        self.n_hidden = n_hidden
        self.shrink = shrink
        self.batch_size = batch_size
        self.patience_ident = patience_ident
        self.history = {}
        self.switch_interval = switch_interval
        self.direction_is_shrink = True
        self.converged_losses = []

        if self.patience_ident is not None: 
            self.patience_shrink = int(np.round(self.patience_ident/4))
        else: 
            self.patience_shrink = 10

        self.patience = patience
        self.n_updates = n_updates
        self.final_netsize = final_netsize
        self.method = method
        self.candidate_perc = candidate_perc

        self.netsize = self.relevant + int(np.ceil((self.n_vars-self.relevant)*self.candidate_perc))

        if self.netsize > self.n_vars:
            print(f"The netsize has been corrected from {self.netsize} to {self.n_vars}.")            
            self.netsize = self.n_vars # correct netsize if necessary 


    
    def run_base_rcn(self,X,y,X_val,y_val):

        ## Preparation ## 
        self.device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu') 

        X, y = X.to(self.device), y.to(self.device)
        if self.patience is not None:
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        self.model = self.model.to(self.device)
        
        # Initialize ident counter
        self.ident_counter = 0

        # initialize shrink counter 
        self.patience_counter_shrink = 0 
        
        # setup the patience and counter for the final_netsize
        if self.final_netsize is not None:
            self.start_netsize = self.netsize
            self.n_steps = min(self.n_updates,10)
            self.patience_shrink = self.n_updates // self.n_steps
            self.direction_is_shrink = self.start_netsize > self.final_netsize
            self.step_nr = 0 

        # initialize stopping flag for shrinking / growing
        self.minimal_netsize = int(np.round(self.relevant * 1.2))
        self.size_is_minimal = self.netsize <= self.minimal_netsize
        self.size_is_maximal = self.netsize == self.n_vars

        # Handle Batch size
        if self.batch_size is None: # Set batch size to full dataset 
            self.batch_size = self.n_train 
        self.batch_size = min(self.batch_size, self.n_train)
                
        # initialize selected randomly
        self.selected = torch.tensor(np.random.choice(self.n_vars, self.netsize, replace=False),device=self.device)
        self.ident_index = torch.tensor(np.random.choice(len(self.selected), self.relevant, replace=False),device=self.device)
        self.ident = self.selected[self.ident_index]
        self.selected_candidate_mask = torch.ones(len(self.selected), dtype=torch.bool,device=self.device) # indicating candidates among selected (here: all)

        # Initialize early stopping flag
        stopearly = None

        # Initialize data structures
        self.highscores = torch.zeros(self.n_vars,device=self.device) -100
        
        # Initialize grads
        grads_temp =  torch.zeros_like(self.model[0]._parameters['weight'],device = self.device) 

        # Initialize step_counter
        step_counter = 0
        break_flag = False

        # Initialize update counter and update_history
        update_counter = 0

        # Initialize patience counter
        # if self.patience is not None: 
        bad_val_counter = 0
        self.best_val_loss = float('inf')

        # initialize old weights
        if self.method == "delta_weight":
            old_weights = self.get_weights()

        ## Training ## 
        for epoch in range(self.epochs):

            if epoch % 200 == 0 :
                print(f'epoch {epoch} started. val {bad_val_counter}, ident {self.ident_counter}')

            ## Training ## 
            indices = torch.randperm(self.n_train)

            for i in range(self.n_train  // self.batch_size): 
                batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]

                outputs = self.model(X[batch_indices[:, None],self.selected])
                loss = self.criterion(outputs, y[batch_indices])
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                step_counter += 1

                grads_temp += self.model[0]._parameters['weight'].grad.detach()*len(batch_indices)

                if step_counter == self.switch_interval:

                    outputs = self.predict_internal(X,tensor_out=True)
                    train_loss = self.criterion(outputs, y).cpu().detach().numpy()

                    if self.patience is not None:
                        outputs = self.predict_internal(X_val,tensor_out=True)
                        val_loss = self.criterion(outputs, y_val).cpu().detach().numpy()

                    self.history[update_counter] = {'epoch': epoch,
                                                    'selected': np.array(self.selected.cpu()).copy(),
                                                    'ident': np.array(self.ident.cpu()).copy(),
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss if self.patience else None
                                                    }

                    # Early Stopping Checks and Counter Updates 
                    if update_counter > 0:
                        
                        # Calculate indicators
                        is_identical = set(self.history[update_counter]['ident']) == set(self.history[update_counter-1]['ident'])

                        if is_identical:
                            self.ident_counter +=1
                        else:
                            self.ident_counter = 0

                        # Early stopping on ident 
                        if (self.patience_ident is not None) and (self.ident_counter >= self.patience_ident):
                            stopearly = 'ident'
                            print(f"Early ident stopping on epoch {epoch} / update {update_counter}")
                            break_flag = True
                            break


                        # Early Stopping on Validation loss
                        if self.patience is not None:

                            if self.history[update_counter]['val_loss']<self.best_val_loss:
                                self.best_val_loss = self.history[update_counter]['val_loss']
                                self.best_update = update_counter
                            
                            if update_counter>self.best_update:
                                bad_val_counter += 1
                                self.patience_counter_shrink += 1
                            else:                        
                                bad_val_counter = 0
                                self.patience_counter_shrink = 0
                              
                            if bad_val_counter >= self.patience or self.ident_counter >= self.patience:
                                if bad_val_counter >= self.patience:
                                    stopearly = 'val'
                                else:
                                    stopearly = "ident"
                                print(f"################# Early {stopearly} stopping on epoch {epoch} / update {update_counter}")
                                break_flag = True
                                break


                    # Incrementing for final netsize:
                    if self.final_netsize is not None:
                        self.patience_counter_shrink += 1


                    # Stopping based on fixed number of updates
                    if self.n_updates is not None:
                        if update_counter >= self.n_updates:
                            print(f"Stopping on epoch {epoch} / update {update_counter}")
                            break_flag = True
                            break

                    if self.method == "weights": 
                        # overwrite grads_temp with the current weights 
                        grads_temp = self.get_weights()

                    if self.method == "delta_weight":
                        grads_temp = self.get_weights() - old_weights

                    self.update_network(grads_temp,epoch,update_counter)
                    grads_temp =  torch.zeros_like(self.model[0]._parameters['weight'],device = self.device) 
                    if self.method == "delta_weight":
                        old_weights = self.get_weights()
                    step_counter = 0
                    update_counter += 1
            if break_flag:
                break

        # Stopping based on epoch
        if epoch == self.epochs - 1 :
            update_counter -= 1 #adjust the update counter
            print(f"Stopping on epoch {epoch} / update {update_counter}")

        # Retrieve ident for update with best loss
        lowest_loss = float('inf')  # Initialize with infinity

        if self.patience:
            for update, data in self.history.items():
                if data['val_loss'] < lowest_loss:
                    lowest_loss = data['val_loss']
                    self.lowest_loss_update = update
        else:
            # Step 1: Find the epoch with the lowest train loss
            for update, data in self.history.items():
                if data['train_loss'] < lowest_loss:
                    lowest_loss = data['train_loss']
                    self.lowest_loss_update = update

        # Retrieve selected and ident from the best epoch
        self.selected_final = self.history[update_counter]['selected']
        self.ident_final = self.history[update_counter]['ident']



    def update_network(self,grads_temp,epoch,update):

        grads_temp /= self.n_train
        grads_temp = grads_temp.abs().sum(dim=0)
        grads_temp = (grads_temp - grads_temp.mean())/(grads_temp.std()) # normalize

        self.highscores[self.selected[self.selected_candidate_mask]] = torch.max(self.highscores[self.selected[self.selected_candidate_mask]], grads_temp[self.selected_candidate_mask])
        _, self.ident_index = torch.topk(self.highscores[self.selected], self.relevant, largest=True)
        self.ident = self.selected[self.ident_index] # update ident

        # update candidate mask
        self.selected_candidate_mask = torch.ones(len(self.selected), dtype=torch.bool,device=self.device)
        self.selected_candidate_mask[self.ident_index] = False

        # Shrink network
        if self.shrink and self.patience_counter_shrink >= self.patience_shrink:

            if self.final_netsize is None:
                # determine the direction (shrink / extend)
                # if train error of this change is higher than last one, change direction 
                last_loss = self.history[update]['val_loss']
                if self.converged_losses and self.converged_losses[-1] < last_loss:
                    self.direction_is_shrink = not self.direction_is_shrink # reverse direction
                self.converged_losses.append(last_loss)
                print(last_loss)

            if self.direction_is_shrink and not self.size_is_minimal:

                if self.final_netsize is not None:
                    self.step_nr += 1 
                    self.netsize = int(np.round(self.final_netsize + (self.start_netsize-self.final_netsize) * ((self.n_steps  - self.step_nr)/self.n_steps)))
                    print(f'step_nr: {self.step_nr}, new netsize: {self.netsize}')
                else: 
                    self.netsize =  np.max([self.minimal_netsize, self.relevant + int(np.round((self.netsize - self.relevant)*0.5))]) #shrink netsize by factor of 2 
                n_to_remove = len(self.selected) - self.netsize

                print(f'reducing netsize from {len(self.selected)} to {self.netsize} on epoch {epoch} / update {update}')

                drop_index = torch.where(self.selected_candidate_mask)[0][:n_to_remove]
                mask_keep = torch.tensor(np.ones(self.selected.shape, dtype=bool),device=self.device) # keep ident and non-dropped others
                mask_keep[drop_index] = False

                # adjust the network
                self.model[0].weight = nn.Parameter(self.model[0].weight[:, mask_keep]) # update weights
                # update optimizer
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

                # update selected
                self.selected = self.selected[mask_keep]
                self.ident_index = (self.selected == self.ident.unsqueeze(1)).any(dim=0)

            if not self.direction_is_shrink and not self.size_is_maximal:

                if self.final_netsize is not None:
                    self.step_nr += 1 
                    self.netsize = int(np.round(self.final_netsize + (self.start_netsize-self.final_netsize) * ((self.n_steps  - self.step_nr)/self.n_steps)))
                    print(f'step_nr: {self.step_nr}, new netsize: {self.netsize}')
                else:
                    self.netsize = np.min([self.n_vars, self.relevant + int(np.round((self.netsize - self.relevant)*2))]) #increase netsize by factor of 2 
                
                print(f'increasing netsize from {len(self.selected)} to {self.netsize} on epoch {epoch} / update {update}')

                current_weights = self.model[0].weight.data
                additional_weights = torch.randn(current_weights.size(0), self.netsize-len(self.selected), device=self.device) # Initialize new weights

                # adjust network
                self.model[0].weight = nn.Parameter(torch.cat((current_weights, additional_weights), dim=1))
                # update optimizer
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

                # update selected
                self.selected = torch.cat((self.selected,  torch.zeros(self.netsize-len(self.selected), dtype=self.selected.dtype, device=self.selected.device)))

            # update candidate mask
            self.selected_candidate_mask = torch.ones(len(self.selected), dtype=torch.bool,device=self.device)
            self.selected_candidate_mask[self.ident_index] = False

            # update counters
            self.patience_counter_shrink = 0 

            self.size_is_minimal = self.netsize <= self.minimal_netsize
            self.size_is_maximal = self.netsize == self.n_vars

        # Update selected
        new_vars_pool = list(set(range(self.n_vars)) - set(self.ident.cpu().numpy()))
        new_vars = torch.tensor(np.random.choice(new_vars_pool, len(self.selected)-self.relevant, replace=False),device=self.device)
        self.selected[self.selected_candidate_mask] = new_vars


        # Re-initialize the parameters of replaced neurons
        stdv = (1e-8)
        self.model[0].weight.data[:, self.selected_candidate_mask] = torch.empty(sum(self.selected_candidate_mask),device = self.device).uniform_(-stdv, stdv)
             

    def get_weights(self):
        return self.model[0]._parameters['weight'].detach().clone()

    def predict_internal(self,X, prob = False, tensor_out=False):
        # predict while setting non-ident values to zero

        mask = torch.zeros_like(X, dtype=bool, device=self.device)
        mask[:, self.ident] = True
        X_masked = torch.where(mask, X, torch.zeros_like(X))
        y_pred = self.model(X_masked[:, self.selected])

        if prob:
            y_pred = nn.functional.softmax(y_pred, dim=1)

        if not tensor_out:
            y_pred = y_pred.cpu().detach().numpy()        

        return y_pred
    
    
    def predict(self,X_test, prob = False, tensor_out=False):

        if not torch.is_tensor(X_test):
            X_test = torch.from_numpy(X_test).float()
        
        model= self.model.to(self.device)
        X_test = X_test.to(self.device)

        mask = torch.zeros_like(X_test, dtype=bool,device = self.device)
        mask[:, self.ident_final] = True
        X_test[~mask] = 0
        X_test = X_test[:, self.selected_final]
        y_pred = model(X_test)

        if prob:
            # apply softmax
            y_pred = nn.functional.softmax(y_pred, dim=1)

        if not tensor_out:
            y_pred = y_pred.cpu().detach().numpy()        

        return y_pred


    def convert_to_tensor(self, X, y, tensor_y_dtype):
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        if not torch.is_tensor(y):
            y = torch.from_numpy(y).type(tensor_y_dtype)
        return X, y


# TODO
# class rcn_reg(BaseRCN):

class rcn_class(BaseRCN):
    def __init__(self, X, y,X_val=None,y_val=None, **kwargs):
        self.n_train = X.shape[0] 
        self.n_vars = X.shape[1]
        super().__init__(**kwargs)
        self.fit(X, y,X_val,y_val)
    
    def fit(self, X, y,X_val,y_val):
        X, y = self.convert_to_tensor(X, y, tensor_y_dtype=torch.long)
        if X_val is not None:
            X_val,y_val = self.convert_to_tensor(X_val, y_val, tensor_y_dtype=torch.long)
        unique_values = torch.unique(y)
        n_classes = len(unique_values)
        layers = [
            nn.Linear(self.netsize, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, n_classes)
        ]
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.run_base_rcn(X, y,X_val,y_val)






