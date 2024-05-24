import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.utils.generic as generic
import rationale_net.utils.metrics as metrics
import tqdm
import numpy as np
import pdb
import sklearn.metrics
import rationale_net.utils.learn as learn
from collections import defaultdict

def train_model(train_loader, valid_loader, model, gen, args):
    '''
    Train model and tune on valid set. If model doesn't improve valid performance within args.patience
    epochs, then halve the learning rate, restore the model to best and continue training.

    At the end of training, the function will restore the model to best valid version.

    returns epoch_stats: a dictionary of epoch level metrics for train and test
    returns model : best model from this call to train
    '''

    if args.cuda:
        model = model.cuda()
        gen = gen.cuda()

    args.lr = args.init_lr
    optimizer = learn.get_optimizer([model, gen], args)

    num_epoch_sans_improvement = 0
    epoch_stats = metrics.init_metrics_dictionary(modes=['train', 'valid'])
    step = 0
    tuning_key = "valid_{}".format(args.tuning_metric)
    best_epoch_func = min if tuning_key == 'loss' else max

    for epoch in range(1, args.epochs + 1):

        print("-------------\nEpoch {}:\n".format(epoch))
        for mode, data_loader in [('Train', train_loader), ('Valid', valid_loader)]:
            train_model = mode == 'Train'
            print('{}'.format(mode))
            key_prefix = mode.lower()
            epoch_details, step, _, _, _, _ = run_epoch(
                data_loader=data_loader,
                train_model=train_model,
                model=model,
                gen=gen,
                optimizer=optimizer,
                step=step,
                args=args)

            epoch_stats, log_statement = metrics.collate_epoch_stat(epoch_stats, epoch_details, key_prefix, args)

            # Log  performance
            print(log_statement)


        # Save model if beats best valid
        best_func = min if args.tuning_metric == 'loss' else max
        if best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            # Subtract one because epoch is 1-indexed and arr is 0-indexed
            epoch_stats['best_epoch'] = epoch - 1
            torch.save(model, args.model_path)
            torch.save(gen, learn.get_gen_path(args.model_path))
        else:
            num_epoch_sans_improvement += 1

        if not train_model:
            print('---- Best Valid {} is {:.4f} at epoch {}'.format(
                args.tuning_metric,
                epoch_stats[tuning_key][epoch_stats['best_epoch']],
                epoch_stats['best_epoch'] + 1))

        if num_epoch_sans_improvement >= args.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            model.cpu()
            gen.cpu()
            model = torch.load(args.model_path)
            gen = torch.load(learn.get_gen_path(args.model_path))

            if args.cuda:
                model = model.cuda()
                gen   = gen.cuda()
            args.lr *= .5
            optimizer = learn.get_optimizer([model, gen], args)

    # Restore model to best valid performance
    if os.path.exists(args.model_path):
        model.cpu()
        model = torch.load(args.model_path)
        gen.cpu()
        gen = torch.load(learn.get_gen_path(args.model_path))

    return epoch_stats, model, gen


def test_model(test_loader, model, gen, args):
    '''
    Run model on test data, and return loss, accuracy.
    '''
    if args.cuda:
        model = model.cuda()
        gen = gen.cuda()

    test_stats = metrics.init_metrics_dictionary(modes=['test'])

    mode = 'Test'
    train_model = False
    key_prefix = mode.lower()
    print("-------------\nTest")
    epoch_details, _, losses, preds, golds, rationales = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model,
        gen=gen,
        optimizer=None,
        step=None,
        args=args)

    test_stats, log_statement = metrics.collate_epoch_stat(test_stats, epoch_details, 'test', args)
    test_stats['losses'] = losses
    test_stats['preds'] = preds
    test_stats['golds'] = golds
    test_stats['rationales'] = rationales

    print(log_statement)

    return test_stats

def run_epoch(data_loader, train_model, model, gen, optimizer, step, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    eval_model = not train_model
    #data_iter = iter(data_loader)

    losses = []
    obj_losses = []
    k_selection_losses = []
    k_continuity_losses = []
    preds = []
    golds = []
    losses = []
    texts = []
    rationales = defaultdict(int)

    if train_model:
        model.train()
        gen.train()
    else:
        gen.eval()
        model.eval()

    num_batches_per_epoch = len(data_loader)
    if train_model:
        num_batches_per_epoch = min(len(data_loader), 10000)

    for i, (inputs, labels) in enumerate(data_loader):
        if train_model:
            step += 1
            if  step % 100 == 0 or args.debug_mode:
                args.gumbel_temprature = max( np.exp((step+1) *-1* args.gumbel_decay), .05)

        x_indx = inputs
        x_indx = x_indx[:, :, None]
        y = labels

        if args.cuda:
            x_indx, y = x_indx.cuda(), y.cuda()

        if train_model:
            optimizer.zero_grad()

        if args.get_rationales:
            mask, z = gen(x_indx)
        else:
            mask = None

        logits = model(x_indx.float(), mask=mask)

        if args.use_as_tagger:
            logits = logits.view(-1, 2)
            y = y.view(-1)

        loss = get_loss(logits, y.long(), args)
        obj_loss = loss

        if args.get_rationales:
            selection_cost = gen.loss(mask, x_indx)
            loss += args.selection_lambda * selection_cost

        if train_model:
            loss.backward()
            optimizer.step()

        if args.get_rationales:
            k_selection_losses.append(selection_cost.detach().numpy())

        obj_losses.append(obj_loss.detach().numpy())
        losses.append(loss.detach().numpy())
        batch_softmax = F.softmax(logits, dim=-1).cpu()
        preds.extend(torch.max(batch_softmax, 1)[1].view(y.size()).data.numpy())
        batch_rationales = learn.get_rationales(mask)
        for feature_list in batch_rationales:
            for feature in feature_list:
                rationales[feature] += 1

        if args.use_as_tagger:
            golds.extend(labels.view(-1).numpy())
        else:
            golds.extend(labels.numpy())

    epoch_metrics = metrics.get_metrics(preds, golds, args)

    epoch_stat = {
        'loss' : np.mean(losses),
        'obj_loss': np.mean(obj_losses)
    }

    for metric_k in epoch_metrics.keys():
        epoch_stat[metric_k] = epoch_metrics[metric_k]

    if args.get_rationales:
        epoch_stat['k_selection_loss'] = np.mean(k_selection_losses)
        
    rationales = dict(rationales)
    rationales = dict(sorted(rationales.items(), key=lambda item: item[1], reverse=True))

    return epoch_stat, step, losses, preds, golds, rationales


def get_loss(logit,y, args):
    if args.objective == 'cross_entropy':
        if args.use_as_tagger:
            loss = F.cross_entropy(logit, y, reduce=False)
            neg_loss = torch.sum(loss * (y == 0).float()) / torch.sum(y == 0).float()
            pos_loss = torch.sum(loss * (y == 1).float()) / torch.sum(y == 1).float()
            loss = args.tag_lambda * neg_loss + (1 - args.tag_lambda) * pos_loss
        else:
            loss = F.cross_entropy(logit, y)
    elif args.objective == 'mse':
        loss = F.mse_loss(logit, y.float())
    else:
        raise Exception(
            "Objective {} not supported!".format(args.objective))
    return loss
