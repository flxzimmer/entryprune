import entryprune

from sklearn.model_selection import train_test_split


def fit_wrapper(stopping,stopping_hyperpar,perc, switch):
    def run(X,y,n_vars_selected):

        args = {
            'epochs': 10e4,
            'lr': 1e-3,
            'n_hidden': 100,
            'batch_size': 1024,
            'switch_interval': switch,
            'candidate_perc': perc,
            'patience_ident' : None, # Ident patience
            'patience' : None, # Validation patience
            'n_updates': None, # for Final run of validation
        }

        if stopping == "ident":
            args['patience_ident'] = stopping_hyperpar
        elif stopping == "epochs":
            args['epochs'] = stopping_hyperpar
        elif stopping == "validation":
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            args['patience'] = stopping_hyperpar
            re = entryprune.ep_class(X=X_train, y=y_train,relevant=n_vars_selected, X_val = X_val, y_val = y_val,**args)
            args['patience'] = None
            args['n_updates'] = re.best_update

        re = entryprune.ep_class(X=X, y=y, relevant=n_vars_selected, **args)
        return re.ident_final
    
    return run



def fit_flex_wrapper(perc, switch):

    def run(X, y,n_vars_selected):

        args = {
            'epochs': 10e4,
            'lr': 1e-3,
            'n_hidden': 100,
            'patience': 100,
            'n_updates' : None,
            'batch_size': 1024,
            'candidate_perc': perc,
            'switch_interval': switch,
            'shrink': True,
            'final_netsize' : None,
        }
    
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        re = entryprune.ep_class(X=X_train, y=y_train, X_val= X_val, y_val = y_val,relevant=n_vars_selected, **args)
        loss, n_updates,final_netsize = re.best_val_loss, re.best_update, len(re.history[re.lowest_loss_update]['selected'])
        args['patience'] = None
        args['n_updates'] = n_updates
        args['final_netsize'] = final_netsize
        re = entryprune.ep_class(X=X, y=y,relevant=n_vars_selected, **args)

        return re.ident_final

    return run

