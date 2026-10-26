import os
import time
import argparse

import torch

from data import create_dataset
from models import create_model
from util.util import mkdirs, print_current_losses


def main(args):
    n_epochs = args.n_epochs
    n_epochs_decay = args.n_epochs_decay
    batch_size = args.batch_size
    gpu_ids = args.gpu_id

    expr_dir = os.path.join(args.checkpoints_dir, args.name)
    mkdirs(expr_dir)
    state_path = os.path.join(expr_dir, 'training_state.pth')
    log_name = os.path.join(expr_dir, 'train_log')

    dataset = create_dataset(
        args.dataroot,
        batch_size=batch_size,
        phase='train',
        load_size=args.load_size,
        crop_size=args.crop_size,
        serial_batches=False,
        input_nc=3,
        output_nc=3,
        no_flip=False,  # enable horizontal flip augmentation
        num_workers=args.num_workers,
    )
    dataset_size = len(dataset)
    print('dataset: %d images (A: %d, B: %d)' % (
        dataset_size, dataset.dataset.A_size, dataset.dataset.B_size))

    model = create_model(
        gpu_ids=gpu_ids,
        isTrain=True,
        checkpoints_dir=args.checkpoints_dir,
        name=args.name,
        continue_train=args.continue_train,
    )
    model.n_epochs = n_epochs
    model.n_epochs_decay = n_epochs_decay
    model.lambda_identity = args.lambda_identity

    if args.pretrained_g_a:
        state = torch.load(args.pretrained_g_a, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        missing, unexpected = model.unwrap_net(model.netG_A).load_state_dict(
            state, strict=False)
        print('loaded pretrained G_A from %s (missing=%d, unexpected=%d)' % (
            args.pretrained_g_a, len(missing), len(unexpected)))

    if args.lr_g_a_scale != 1.0:
        # Fine-tuning a pretrained G_A needs a gentler LR than the randomly
        # initialized G_B; keep them in separate param groups so the scheduler
        # scales both proportionally.
        model.optimizer_G = torch.optim.Adam([
            {'params': model.netG_A.parameters(), 'lr': model.lr * args.lr_g_a_scale},
            {'params': model.netG_B.parameters(), 'lr': model.lr},
        ], lr=model.lr, betas=(model.beta1, 0.999), fused=len(model.gpu_ids) > 0)
        model.optimizers[0] = model.optimizer_G
        print('G_A LR scale: %.3f (G_A lr=%.2e, G_B lr=%.2e)' % (
            args.lr_g_a_scale, model.lr * args.lr_g_a_scale, model.lr))

    epoch_count = 1
    total_iters = 0
    if args.continue_train:
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu')
            epoch_count = int(state.get('epoch', 0)) + 1
            total_iters = int(state.get('total_iters', 0))
            model.resume_epoch = epoch_count - 1
            print('resuming from epoch %d, total_iters %d' % (epoch_count, total_iters))
        else:
            print('continue_train requested but %s not found; starting from scratch' % state_path)

    if not args.no_amp:
        model.enable_amp()
    if not args.no_compile:
        model.enable_compile()

    model.setup()

    print('training config: batch_size=%d, workers=%d, amp=%s, compile=%s, load_size=%d, crop_size=%d'
          % (batch_size, args.num_workers, not args.no_amp, not args.no_compile,
             args.load_size, args.crop_size))
    if not args.no_compile:
        print('note: the first iterations include torch.compile warmup and will be slow')

    for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1):
        if args.freeze_g_a_epochs > 0:
            should_freeze = epoch <= args.freeze_g_a_epochs
            if should_freeze != getattr(model, '_g_a_frozen', False):
                for param in model.netG_A.parameters():
                    param.requires_grad = not should_freeze
                model._g_a_frozen = should_freeze
                print('G_A %s at epoch %d' % (
                    'frozen' if should_freeze else 'unfrozen', epoch))
        if (args.identity_off_after > 0 and epoch > args.identity_off_after
                and model.lambda_identity != 0.0):
            model.lambda_identity = 0.0
            print('identity loss disabled from epoch %d' % epoch)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % args.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += batch_size
            epoch_iter += batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % args.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / batch_size
                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, log_name=log_name)

            iter_data_time = time.time()

        if epoch % args.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            torch.save({'epoch': epoch, 'total_iters': total_iters}, state_path)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', default='./datasets/chinesepainting',
                        help='directory of training data', type=str)
    parser.add_argument('--n_epochs', default=200, help='number of epochs', type=int)
    parser.add_argument('--n_epochs_decay', default=200,
                        help='number of epochs where learning rate decays', type=int)
    parser.add_argument('--batch_size', default=1, help='batch size', type=int)
    parser.add_argument('--num_workers', default=8, help='data loader workers', type=int)
    parser.add_argument('--load_size', default=286, help='resize size before crop', type=int)
    parser.add_argument('--crop_size', default=256, help='training crop size', type=int)
    parser.add_argument('--gpu_id', default='0', help='gpu_id, -1 if use cpu', type=str)
    parser.add_argument('--checkpoints_dir', default='./checkpoints',
                        help='directory to save checkpoints', type=str)
    parser.add_argument('--name', default='chinesepainting_cyclegan',
                        help='name of experiment', type=str)
    parser.add_argument('--print_freq', default=100, help='log frequency in iterations', type=int)
    parser.add_argument('--save_epoch_freq', default=1, help='checkpoint frequency in epochs', type=int)
    parser.add_argument('--continue_train', action='store_true',
                        help='resume from latest checkpoint and training_state.pth')
    parser.add_argument('--no_amp', action='store_true', help='disable fp16 mixed precision')
    parser.add_argument('--no_compile', action='store_true', help='disable torch.compile')
    parser.add_argument('--lambda_identity', default=0.5, type=float,
                        help='weight for identity loss; 0 disables the identity pass')
    parser.add_argument('--identity_off_after', default=0, type=int,
                        help='disable identity loss after this epoch (0 keeps it for the whole run)')
    parser.add_argument('--pretrained_g_a', default='', type=str,
                        help='optional path to a pretrained generator G_A state_dict')
    parser.add_argument('--lr_g_a_scale', default=1.0, type=float,
                        help='LR multiplier for a pretrained G_A (e.g. 0.1 when fine-tuning)')
    parser.add_argument('--freeze_g_a_epochs', default=0, type=int,
                        help='freeze G_A for the first N epochs while G_B/D warm up')

    main(parser.parse_args())
