import argparse
from pprint import pprint

# List of available languages
available_languages = "aswiki|bnwiki|guwiki|hiwiki|knwiki|mlwiki|mrwiki|orwiki|pawiki|tawiki|tewiki|enwiki|all_langwiki".replace('wiki','').split('|')

# List of available loss functions
available_losses = ['mc', 'bce','cbc','rac','dac']

#List of available models
available_backbone_models =['bert-base-multilingual-cased']

#List of aviailable models
available_models = ['naive','hierarchial','graph']

#List of available graph initializations
available_graph_init = ['mbert','random']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Choose languages, batch size, and loss function.')

    # Add a mutually exclusive group to enforce selection of at least one language
    language_group = parser.add_mutually_exclusive_group(required=True)

    for lang in available_languages:
        language_group.add_argument('--' + lang.lower(), dest='languages', action='append_const', const=lang, help=f'Select {lang}')

    # Run configuration
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fp16', action='store_true', help='Use fp16') #TODO: Add fp16 support
    parser.add_argument('--do_train', action='store_true', help='Do training')
    parser.add_argument('--do_eval', action='store_true', help='Do evaluation')
    parser.add_argument('--do_predict', action='store_true', help='Do prediction')
    parser.add_argument('--train_steps', type=int, default=-1, help='Total number of steps.Takes precednece over number of epochs')
    parser.add_argument('--logging_steps', type=int, default=500, help='Logging steps')
    parser.add_argument('--saving_steps', type=int, default=10000, help='Save steps')
    parser.add_argument('--save_results', action='store_true', help='Save results')#TODO: Add save results support
    parser.add_argument('--save_model', action='store_true', help='Save model')#TODO: Add save model support

    # Evaulate Configuration
    parser.add_argument('--train_test_split', type=float, default=0.8, help='Train test split')
    parser.add_argument('--eval_steps', type=int, default=-1, help='Number of evaluation steps')
    parser.add_argument('--eval_num_steps', type=int, default=8, help='Evaulate after every n training steps')

    # Model Configuration
    parser.add_argument('--model_type', choices=available_models, default='naive', help='Model type to use')
    parser.add_argument('--model_name', choices=available_backbone_models, default='bert-base-multilingual-cased', help='Model name to load from huggingface')
    parser.add_argument('--model_path', default='', help='Path to load the model from') 
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')

    
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')

    #Head Configuration
    parser.add_argument('--graph_dim', type=int, default=1024, help='Model dimension')
    parser.add_argument('--init_label_nodes', choices=available_graph_init, default='mbert', help='Number of layers')

    # Data Configuration
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')

    #Training Hyperparameters
    parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--max_chunks', type=int, default=8, help='Maximum number of chunks')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for Evaluation')
    parser.add_argument('--loss_type', choices=available_losses, default='bce', help='Loss function to use')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--correct_bias', action='store_true', help='Correct bias')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--warmup_portion', type=float, default=0.1, help='Logging steps') 
    
    # Saving Paths and data paths
    parser.add_argument('--save_model_path', default='outputs/', help='Path to save the model to')
    parser.add_argument('--data', default='data/', help='Path to save the model to')
    parser.add_argument('--predict_file', default='', help='File to run model on and save_predictions')
    parser.add_argument('--save_predictions', default='./', help='Save predictions')


    
    
    

    
    


    return parser.parse_args()

def get_args():
    args = parse_arguments()

    # Set the selected languages
    selected_languages = available_languages[:-1] if 'all_lang' in args.languages else args.languages
    args.selected_languages = selected_languages

    # Set number of epochs
    if args.train_steps > 0:
        args.num_train_epochs = 1_000_000_000
    else:
        args.train_steps = 1_000_000_000

    if args.eval_steps<0:
        args.eval_steps = 1_000_000_000

    if args.eval_num_steps<0:
        args.eval_num_steps = 1_000_000_000
    assert args.d_model%args.nhead==0, "d_model should be divisible by nhead"
    if args.model_type=='graph':
        assert  args.batch_size == 1, "Batch size should be 1 for graph model"
        assert  args.eval_batch_size == 1, "Batch size should be 1 for graph model"
    
    print('Arguments: ')
    pprint(vars(args))
    return args

    # batch_size = args.batch_size
    # selected_loss = args.loss

    # print('Selected languages:', selected_languages)
    # print('Batch size:', batch_size)
    # print('Selected loss:', selected_loss)

if __name__ == '__main__':
    get_args()
