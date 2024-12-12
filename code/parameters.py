
import torch
import argparse
import sys
class Setting:
    
    def parse(self):
        guess_nyc = any(['NewYork' in argv for argv in sys.argv])
        guess_tky = any(['Tokyo' in argv for argv in sys.argv])  
        guess_ist = any(['Istanbul' in argv for argv in sys.argv]) 
        parser = argparse.ArgumentParser()
        self.parse_arguments(parser)    
        if guess_nyc:
            self.parse_NewYork(parser)
        elif guess_tky:
            self.parse_Tokyo(parser)
        elif guess_ist:              
            self.parse_Istanbul(parser)
        else:
            print('Using default parameters for NewYork dataset')
            self.parse_NewYork(parser)
        args = parser.parse_args()

        self.dataset_name = args.dataset_name
        self.tpp_context_size = args.tpp_context_size 
        self.tpp_mark_embedding_size = args.tpp_mark_embedding_size  
        self.tpp_time_embedding_size = args.tpp_time_embedding_size
        self.time_interval_embedding_size = args.time_interval_embedding_size
        self.tpp_loc_embedding_size = args.tpp_loc_embedding_size
        self.tpp_num_mix_components = args.tpp_num_mix_components
        self.tpp_sequential_type = args.tpp_sequential_type
        self.batch_size = args.batch_size
        self.regularization = args.regularization
        self.learning_rate = args.learning_rate
        self.display_step = args.display_step
        self.seed = args.seed
        self.experiment_comments = args.experiment_comments
        self.max_epochs = args.max_epochs
        self.tpp_sequence_embedding_size = args.tpp_sequence_embedding_size
        self.min_seq_len = args.min_seq_len
        self.vae_hidden = args.vae_hidden
        self.vae_latent = args.vae_latent
        self.vae_batch_size = args.vae_batch_size
        self.vae_mse_lambda = args.vae_mse_lambda
        self.vae_epochs = args.vae_epochs

    def parse_arguments(self, parser):
        parser.add_argument("--dataset_name", type=str, default="NewYork.pkl", help="dataset")
        parser.add_argument("--tpp_context_size", type=int, default=64, help="Size of the sequential hidden vector")
        parser.add_argument("--tpp_mark_embedding_size", type=int, default=64, help="Size of the mark embedding (used as sequential input)")
        parser.add_argument("--tpp_time_embedding_size", type=int, default=64, help="Size of the mark embedding (used as sequential input)")
        parser.add_argument("--time_interval_embedding_size", type=int, default=64, help="Size of the time-interval embedding (used as sequential input)")
        parser.add_argument("--tpp_loc_embedding_size", type=int, default=64, help="Size of the loc embedding (used as sequential input)")
        parser.add_argument("--tpp_num_mix_components", type=int, default=64, help="Number of components for a mixture model")
        parser.add_argument("--tpp_sequential_type", type=str, default="GRU", help="encoder {RNN, GRU, LSTM}")
        parser.add_argument("--regularization", type=float, default=1e-5, help="L2 regularization parameter")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam optimizer")
        parser.add_argument("--display_step", type=int, default=1, help="Display training statistics after every display_step")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--experiment_comments", type=str, default=None, help="Can manually set generated name for different run")


    def parse_NewYork(self, parser):
        parser.add_argument("--batch_size", type=int, default=16, help="Number of sequences in a batch")
        parser.add_argument("--max_epochs", type=int, default=50, help="For how many epochs to train")
        parser.add_argument("--tpp_sequence_embedding_size", type=int, default=32, help="Size of the mark embedding (used as sequential input)")
        parser.add_argument("--min_seq_len", type=int, default=5)
        parser.add_argument("--vae_hidden", type=int, default=16)
        parser.add_argument("--vae_latent", type=int, default=2)
        parser.add_argument("--vae_batch_size", type=int, default=64)
        parser.add_argument("--vae_mse_lambda", type=float, default=1.2)
        parser.add_argument("--vae_epochs", type=int, default=5)


    def parse_Tokyo(self, parser):
        parser.add_argument("--batch_size", type=int, default=16, help="Number of sequences in a batch")
        parser.add_argument("--max_epochs", type=int, default=60, help="For how many epochs to train")
        parser.add_argument("--tpp_sequence_embedding_size", type=int, default=16, help="Size of the mark embedding (used as sequential input)")
        parser.add_argument("--min_seq_len", type=int, default=10)
        parser.add_argument("--vae_hidden", type=int, default=8)
        parser.add_argument("--vae_latent", type=int, default=2)
        parser.add_argument("--vae_batch_size", type=int, default=32)
        parser.add_argument("--vae_mse_lambda", type=float, default=6)
        parser.add_argument("--vae_epochs", type=int, default=5)
    
    def parse_Istanbul(self, parser):
        parser.add_argument("--batch_size", type=int, default=16, help="Number of sequences in a batch")
        parser.add_argument("--max_epochs", type=int, default=40, help="For how many epochs to train")
        parser.add_argument("--tpp_sequence_embedding_size", type=int, default=32, help="Size of the mark embedding (used as sequential input)")
        parser.add_argument("--min_seq_len", type=int, default=5)
        parser.add_argument("--vae_hidden", type=int, default=16)
        parser.add_argument("--vae_latent", type=int, default=2)
        parser.add_argument("--vae_batch_size", type=int, default=64)
        parser.add_argument("--vae_mse_lambda", type=float, default=1)
        parser.add_argument("--vae_epochs", type=int, default=5)

