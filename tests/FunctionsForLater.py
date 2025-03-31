    def get_params(self, deep=True):
        return {
            "hidden_size": self.hidden_size,
            "init_method": self.init_method,
            "kohonen_lr": self.kohonen_lr,
            "grossberg_lr": self.grossberg_lr,
            "max_epochs": self.max_epochs,
            "neighborhood_size": self.neighborhood_size,
            "batch_size": self.batch_size,
            "use_autoencoder": self.use_autoencoder,
            "ae_dim": self.ae_dim,
            "ae_epochs": self.ae_epochs,
            "ae_lr": self.ae_lr,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "early_stopping_no_improve_patience": self.early_stopping_no_improve_patience,
            "early_stopping_error_increase_patience": self.early_stopping_error_increase_patience,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "device": self.device.type if self.device else None,
            "log_interval": self.log_interval,
            "ae_hidden_layers": self.ae_hidden_layers,
            "ae_activation": self.ae_activation,
            "use_ae_conv": self.use_ae_conv,
            "distance_metric": self.distance_metric,
            "neighborhood_function": self.neighborhood_function,
            "kohonen_lr_scheduler": self.kohonen_lr_scheduler,
            "grossberg_lr_scheduler": self.grossberg_lr_scheduler,
            "ae_lr_scheduler": self.ae_lr_scheduler
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save(self, filepath):
        torch.save({
            'cpnn_state_dict': self.cpnn_.state_dict(),
            'autoencoder_state_dict': None if self.autoencoder_ is None else self.autoencoder_.state_dict(),
            'hyperparameters': self.get_params(),
            'classes': self.classes_,
            'input_size': self.input_size_
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.input_size_ = checkpoint['input_size']
        self.classes_ = checkpoint['classes']

        self.cpnn_ = CounterPropagationNetwork(self.input_size_,
                                                checkpoint['hyperparameters']['hidden_size'],
                                                len(self.classes_),
                                                init_method=checkpoint['hyperparameters']['init_method']).to(self.device)
        self.cpnn_.load_state_dict(checkpoint['cpnn_state_dict'])

        if checkpoint['autoencoder_state_dict'] is not None:
            self.autoencoder_ = Autoencoder(self.input_size_,
                                            checkpoint['hyperparameters']['ae_dim'],
                                            hidden_layers=checkpoint['hyperparameters']['ae_hidden_layers'],
                                            activation=checkpoint['hyperparameters']['ae_activation'],
                                            conv=checkpoint['hyperparameters']['use_ae_conv']).to(self.device)
            self.autoencoder_.load_state_dict(checkpoint['autoencoder_state_dict'])
        else:
            self.autoencoder_ = None

        loaded_params = checkpoint['hyperparameters']
        loaded_params['kohonen_lr_scheduler'] = None
        loaded_params['grossberg_lr_scheduler'] = None
        loaded_params['ae_lr_scheduler'] = None
        self.set_params(**loaded_params)

        logging.info(f"Model loaded from {filepath}")
        return self