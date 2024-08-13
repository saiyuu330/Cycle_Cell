import os
import torch.nn as nn
from torch.optim import Adam
import itertools
from .base import *
from model import Generator, Discriminator
from torch.optim.lr_scheduler import LambdaLR
from data.dataset import create_cycle_loader, augment_dataset


def lambda_rule(epoch):
    decay_epoch = 20

    def lr_lambda(current_epoch):
        return 1.0 - max(0, current_epoch - decay_epoch) / float(decay_epoch)
    return lr_lambda


def save_checkpoint(epoch, generator_G, generator_F, discriminator_D_A, discriminator_D_B,
                    optimizer_G, optimizer_D_A, optimizer_D_B, checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save({
        'epoch': epoch,
        'generator_G_state_dict': generator_G.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
    }, os.path.join(checkpoint_dir, f'generator_A_to_B_{epoch}.pth'))

    torch.save({
        'epoch': epoch,
        'generator_F_state_dict': generator_F.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
    }, os.path.join(checkpoint_dir, f'generator_B_to_A_{epoch}.pth'))

    torch.save({
        'epoch': epoch,
        'discriminator_D_A_state_dict': discriminator_D_A.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
    }, os.path.join(checkpoint_dir, f'discriminator_D_A_{epoch}.pth'))

    torch.save({
        'epoch': epoch,
        'discriminator_D_B_state_dict': discriminator_D_B.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
    }, os.path.join(checkpoint_dir, f'discriminator_D_B_{epoch}.pth'))

    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(epoch, generator_G, generator_F, discriminator_D_A, discriminator_D_B,
                    optimizer_G, optimizer_D_A, optimizer_D_B, checkpoint_dir='checkpoints'):
    generator_G_path = os.path.join(checkpoint_dir, f'generator_A_to_B_{epoch}.pth')
    generator_F_path = os.path.join(checkpoint_dir, f'generator_B_to A_{epoch}.pth')
    discriminator_D_A_path = os.path.join(checkpoint_dir, f'discriminator_D_A_{epoch}.pth')
    discriminator_D_B_path = os.path.join(checkpoint_dir, f'discriminator_D_B_{epoch}.pth')

    if os.path.isfile(generator_G_path):
        checkpoint = torch.load(generator_G_path)
        generator_G.load_state_dict(checkpoint['generator_G_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        print(f'Loaded checkpoint for generator_A_to_B from epoch {epoch}')
    if os.path.isfile(generator_F_path):
        checkpoint = torch.load(generator_F_path)
        generator_F.load_state_dict(checkpoint['generator_F_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        print(f'Loaded checkpoint for generator_B_to_A from epoch {epoch}')
    if os.path.isfile(discriminator_D_A_path):
        checkpoint = torch.load(discriminator_D_A_path)
        discriminator_D_A.load_state_dict(checkpoint['discriminator_D_A_state_dict'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
        print(f'Loaded checkpoint for discriminator_D_A from epoch {epoch}')
    if os.path.isfile(discriminator_D_B_path):
        checkpoint = torch.load(discriminator_D_B_path)
        discriminator_D_B.load_state_dict(checkpoint['discriminator_D_B_state_dict'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
        print(f'Loaded checkpoint for discriminator_D_B from epoch {epoch}')


class Trainer(Learner):
    def __init__(self, args):
        super().__init__(args)
        self.G = Generator().to(args.device)  # A -> B
        self.F = Generator().to(args.device)  # B -> A
        self.D_A = Discriminator().to(args.device)  # A 판별기
        self.D_B = Discriminator().to(args.device)  # B 판별기

        # 손실 함수
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # 최적화 설정
        self.optimizer_G = Adam(itertools.chain(self.G.parameters(), self.F.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_A = Adam(self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_B = Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.model_state = None
        self.optimizer = None
        self.device = args.device

    def train(self):
        set_seed(self.seed)
        loss_list = []

        checkpoint_dir = './content/drive/MyDrive/checkpoints'
        start_epoch = 0
        load_checkpoint(start_epoch, self.G, self.F, self.D_A, self.D_B,
                        self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B, checkpoint_dir)

        scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=lambda_rule(self.epochs))
        scheduler_D_A = LambdaLR(self.optimizer_D_A, lr_lambda=lambda_rule(self.epochs))
        scheduler_D_B = LambdaLR(self.optimizer_D_B, lr_lambda=lambda_rule(self.epochs))

        path_a = os.path.join(self.data_dir, "01. source data")
        path_b = os.path.join(self.data_dir, "02. target data")
        path_ap = './image/03. augmented source data'
        if not os.path.isdir(path_ap):
            augment_dataset(path_a, path_ap, len(os.listdir(path_b)), self.img_size)
            path_a = path_ap
        print("====================== data augment done. ======================")

        dataloader_A = create_cycle_loader(path_a, self.batch_size, self.is_train, self.img_size)

        dataloader_B = create_cycle_loader(path_b, self.batch_size, self.is_train, self.img_size)
        print("====================== make dataloader done. ======================")

        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss_G, epoch_loss_D_A, epoch_loss_D_B = 0.0, 0.0, 0.0
            num_batches = 0

            for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):

                num = int(self.img_size/8 - 6/4)
                valid = torch.ones((real_A.size(0), 1, num, num), device=self.device)  # 진짜일 경우의 판별기 출력
                fake = torch.zeros((real_A.size(0), 1, num, num), device=self.device)  # 가짜일 경우의 판별기 출력

                # ----------------------
                #  Generator 학습
                # ----------------------

                self.optimizer_G.zero_grad()

                real_A = real_A.to(self.device)
                fake_B = self.G(real_A)
                pred_fake = self.D_B(fake_B)
                loss_GAN_AB = self.criterion_GAN(pred_fake, valid)

                real_B = real_B.to(self.device)
                fake_A = self.F(real_B)
                pred_fake = self.D_A(fake_A)
                loss_GAN_BA = self.criterion_GAN(pred_fake, valid)

                recov_A = self.F(fake_B)
                loss_cycle_A = self.criterion_cycle(recov_A, real_A)

                recov_B = self.G(fake_A)
                loss_cycle_B = self.criterion_cycle(recov_B, real_B)

                loss_G = (loss_GAN_AB + loss_GAN_BA) + 10 * (loss_cycle_A + loss_cycle_B)

                loss_G.backward()
                self.optimizer_G.step()

                # ----------------------
                #  Discriminator 학습
                # ----------------------

                # 도메인 A 판별기 학습
                self.optimizer_D_A.zero_grad()

                pred_real = self.D_A(real_A)
                loss_D_real = self.criterion_GAN(pred_real, valid)

                pred_fake = self.D_A(fake_A.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, fake)

                loss_D_A = (loss_D_real + loss_D_fake) / 2
                loss_D_A.backward()
                self.optimizer_D_A.step()

                # 도메인 B 판별기 학습
                self.optimizer_D_B.zero_grad()

                pred_real = self.D_B(real_B)
                loss_D_real = self.criterion_GAN(pred_real, valid)

                pred_fake = self.D_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, fake)

                loss_D_B = (loss_D_real + loss_D_fake) / 2
                loss_D_B.backward()
                self.optimizer_D_B.step()

                epoch_loss_G += loss_G.item()
                epoch_loss_D_A += loss_D_A.item()
                epoch_loss_D_B += loss_D_B.item()
                num_batches += 1

            avg_loss_G = epoch_loss_G / num_batches
            avg_loss_D_A = epoch_loss_D_A / num_batches
            avg_loss_D_B = epoch_loss_D_B / num_batches

            print(f"[Epoch {epoch + 1}/{self.epochs}] "
                  f"[D_A loss: {avg_loss_D_A:.4f}] [D_B loss: {avg_loss_D_B:.4f}] "
                  f"[G loss: {avg_loss_G:.4f}]")

            loss_list.append([avg_loss_D_A, avg_loss_D_B, avg_loss_G])

            scheduler_G.step()
            scheduler_D_A.step()
            scheduler_D_B.step()

            save_interval = 10
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(epoch + 1, self.G, self.F, self.D_A, self.D_B,
                                self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B, checkpoint_dir)

        return loss_list
