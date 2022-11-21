import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from data import TetrominoesDataset
from model import SlotAttentionModel

class Trainer:

    def __init__(self, model, optimizer, device):

        self.model  = model
        self.optimizer = optimizer
        self.loss = nn.MSELoss()
        self.device = device
        self.model.to(self.device)
        self.step =0

    def train_step(self, inputs):

        inputs = inputs.to(self.device)
        recon_combined, recon, masks, slots = self.model(inputs)
        loss = self.loss(recon_combined, inputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def test_step(self, inputs):

        inputs = inputs.to(self.device)
        with torch.no_grad():
            recon_combined, recon, masks, slots = self.model(inputs)
            loss = self.loss(recon_combined, inputs)
        return loss

    def visualize(self, inputs):

        inputs = inputs.to(self.device)
        with torch.no_grad():
            recon_combined, recon, masks, slots = self.model(inputs)

        out = torch.cat(
                [
                    inputs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recon * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        out = (out * 0.5 + 0.5).clamp(0, 1)
        batch_size, num_slots, C, H, W = recon.shape
        images = make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def save(self, dir_path):

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step
            }, os.path.join(dir_path, f'ckpt_{self.step}.pt'))

    def restore(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            state_dict = torch.load(f)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.step = state_dict['step']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--use_implicit_grads", action='store_true')
    parser.add_argument("--n_samples", type=int, default=4, help='number of sample imgs to visualize')
    parser.add_argument("--data_path",  type=str, default='Tetrominoes')
    parser.add_argument("--train_h5_path", type=str, default='tetrominoes.h5')
    parser.add_argument("--test_h5_path", type=str, default='tetrominoes_test.h5')
    parser.add_argument("--exp_name", type=str, default='slot_attention', help='name of experiment')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_slot_iters", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Steps for the learning rate decay.')
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--ckpt_interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--restore", action='store_true')
    parser.add_argument("--restore_path", type=str, help='checkpoint path to restore')
  
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')

    if not (os.path.exists(results_dir)):
        os.makedirs(results_dir)

    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(results_dir, logdir)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    train_dataset = TetrominoesDataset(data_h5_path = args.train_h5_path, )
    test_dataset = TetrominoesDataset(data_h5_path = args.test_h5_path, masks=True, d_set='test')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True
                              )

    test_loader   = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True)

    model = SlotAttentionModel(
                    resolution=(35, 35),
                    num_slots = args.num_slots,
                    num_iters = args.num_slot_iters,
                    device= device,
                    in_channels =3,
                    num_hidden = 4,
                    hdim = 32,
                    slot_size = 64,
                    slot_mlp_size = 128,
                    decoder_resolution=(35, 35),
                    implicit_grads = args.use_implicit_grads)
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    trainer = Trainer(model, optimizer, device)
    writer = SummaryWriter(logdir)

    perm = torch.randperm(args.batch_size)
    idx = perm[: args.n_samples]
    fixed_imgs = (next(iter(test_loader))['imgs'])[idx]  # for visualizing slots during training

    if args.train:

        print("Training")
        start_time = time.time()
        step=0
        if args.restore:
            trainer.restore(args.restore_path)
            if trainer.step !=0:
                step = trainer.step + 1
        
        while step <= args.max_steps:

            for batch in train_loader:

                trainer.model.train()
                if step < args.warmup_steps:
                    learning_rate = args.learning_rate * (step/ args.warmup_steps)
                else:
                    learning_rate = args.learning_rate

                learning_rate = learning_rate * (args.decay_rate ** (
                    step / args.decay_steps))

                trainer.optimizer.param_groups[0]['lr'] = learning_rate

                loss = trainer.train_step(batch['imgs'])

                if step % args.log_interval == 0:
                    writer.add_scalar('train_loss', loss.item(), step)
                    trainer.model.eval()
                    sample_imgs = trainer.visualize(fixed_imgs)
                    writer.add_image(f'slots at epoch {step}', sample_imgs, step)
                    save_image(sample_imgs, os.path.join(logdir, f'slots_at_{step}.jpg'))

                    total_loss = 0
                    for batch in test_loader:
                        loss = trainer.test_step(batch['imgs'])
                        total_loss += loss.item()
                    test_loss = total_loss/len(test_loader)

                    writer.add_scalar('test_loss',test_loss, step)

                    print("###############################")
                    print(f"At training step {step}")
                    print("###############################")
                    print(f"Train_loss: {loss.item():.6f}")
                    print(f"test_loss: {test_loss:.6f}")   

                    time_since_start = time.time() - start_time
                    print(f"Time Since Start {time_since_start:.6f}")

                if step % args.ckpt_interval == 0:
                    trainer.save(logdir)

                step+=1
                trainer.step = step
         
    if args.test:
        print("Testing")
        trainer.model.eval()
        total_loss = 0
        for batch in test_loader:
            loss = trainer.test_step(batch['imgs'])
            total_loss += loss.item()
        test_loss = total_loss/len(test_loader)
        print(f"Val_loss: {test_loss:.6f}")

    if args.visualize:
        print("Visualize Slots")
        trainer.model.eval()
        perm = torch.randperm(args.batch_size)
        idx = perm[: args.n_samples]
        batch = next(iter(test_loader))['imgs'][idx]
        images = trainer.visualize(batch)
        save_image(images, os.path.join(logdir, f'slots_at_test.jpg'))
        

if __name__ == '__main__':
    main()
