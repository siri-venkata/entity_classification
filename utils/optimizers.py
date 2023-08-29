import torch

from transformers import AdamW, get_linear_schedule_with_warmup

def get_optimizer_and_scheduler(model,train_size,args):
    """
    Get an AdamW optimizer and a linear learning rate scheduler for fine-tuning a BERT model.
    
    Args:
        model: The BERT model to be fine-tuned.
        learning_rate: The initial learning rate for the optimizer.
        total_steps: The total number of training steps.
        warmup_steps: The number of warmup steps for the learning rate scheduler.

    Returns:
        optimizer: AdamW optimizer.
        scheduler: Linear learning rate scheduler.
    """
    # Get all the named parameters that require gradients
    learning_rate, betas, epsilon,weight_decay,correct_bias, total_steps, warmup_portion = [args.learning_rate,
                                                                                           [args.adam_beta1,args.adam_beta2],
                                                                                             args.adam_epsilon,
                                                                                               args.weight_decay,
                                                                                                 args.correct_bias,
                                                                                                   args.train_steps,
                                                                                                     args.warmup_portion]
    warmup_steps = int(total_steps * warmup_portion)
    param_optimizer = list(model.named_parameters())

    # Separate parameters for weight decay and no weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    # Create the optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon, betas=betas, correct_bias=correct_bias)
    
    if args.model_path:
        try:
            checkpoint = torch.load(args.mode_path+'/optimizer.pt')
            optimizer.load_state_dict(checkpoint)
        except:
            print('No optimizer checkpoint found. Initializing optimizer from scratch.')
    # Create the linear learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(warmup_steps,int(0.1*train_size)), num_training_steps=min(total_steps,train_size*args.num_train_epochs))
    if args.model_path:
        try:
            checkpoint = torch.load(args.mode_path+'/scheduler.pt')
            scheduler.load_state_dict(checkpoint)
        except:
            print('No scheduler checkpoint found. Initializing scheduler from scratch.')
    
    return optimizer, scheduler

    