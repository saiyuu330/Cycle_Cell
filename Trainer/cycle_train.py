import os
import torch.nn as nn
from torch.optim import Adam
import itertools
from .base import *
from model import Generator, Discriminator
from torch.optim.lr_scheduler import LambdaLR
from data.dataset import create_dataloader
from tqdm import trange


def lambda_rule(epoch):
    decay_epoch = 50

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
    generator_G_path = os.path.join(checkpoint_dir, f'generator_G_{epoch}.pth')
    generator_F_path = os.path.join(checkpoint_dir, f'generator_F_{epoch}.pth')
    discriminator_D_A_path = os.path.join(checkpoint_dir, f'discriminator_D_A_{epoch}.pth')
    discriminator_D_B_path = os.path.join(checkpoint_dir, f'discriminator_D_B_{epoch}.pth')

    if os.path.isfile(generator_G_path):
        checkpoint = torch.load(generator_G_path)
        generator_G.load_state_dict(checkpoint['generator_G_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        print(f'Loaded checkpoint for generator_G from epoch {epoch}')
    if os.path.isfile(generator_F_path):
        checkpoint = torch.load(generator_F_path)
        generator_F.load_state_dict(checkpoint['generator_F_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        print(f'Loaded checkpoint for generator_F from epoch {epoch}')
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
        self.G = Generator()  # A -> B
        self.F = Generator()  # B -> A
        self.D_A = Discriminator()  # A 판별기
        self.D_B = Discriminator()  # B 판별기

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

    def train(self):
        set_seed(self.seed)

        checkpoint_dir = 'checkpoints'
        start_epoch = 0
        load_checkpoint(start_epoch, self.G, self.F, self.D_A, self.D_B,
                        self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B, checkpoint_dir)

        lambda_cyc = 10
        lambda_id = 0.5 * lambda_cyc

        scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=lambda_rule(self.epochs))
        scheduler_D_A = LambdaLR(self.optimizer_D_A, lr_lambda=lambda_rule(self.epochs))
        scheduler_D_B = LambdaLR(self.optimizer_D_B, lr_lambda=lambda_rule(self.epochs))

        labels = os.listdir(self.data_dir)
        path_a = os.path.join(self.data_dir, labels[0])
        path_b = os.path.join(self.data_dir, labels[1])

        dataloader_A = create_dataloader(path_a, self.batch_size, self.is_train)
        dataloader_B = create_dataloader(path_b, self.batch_size, self.is_train)

        # 학습 루프
        for epoch in trange(self.epochs):
            for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
                # 진짜 및 가짜 타겟
                valid = torch.ones((real_A.size(0), 1, 10, 10))  # 진짜일 경우의 판별기 출력
                fake = torch.zeros((real_A.size(0), 1, 10, 10))  # 가짜일 경우의 판별기 출력

                # ----------------------
                #  Generator 학습
                # ----------------------

                self.optimizer_G.zero_grad()

                # 도메인 A의 이미지 -> 도메인 B의 이미지
                fake_B = self.G(real_A)
                pred_fake = self.D_B(fake_B)
                loss_GAN_AB = self.criterion_GAN(pred_fake, valid)  # G가 만들어낸 B 이미지의 GAN 손실

                # 도메인 B의 이미지 -> 도메인 A의 이미지
                fake_A = self.F(real_B)
                pred_fake = self.D_A(fake_A)
                loss_GAN_BA = self.criterion_GAN(pred_fake, valid)  # F가 만들어낸 A 이미지의 GAN 손실

                # 순환 일관성 손실
                recov_A = self.F(fake_B)
                loss_cycle_A = self.criterion_cycle(recov_A, real_A)

                recov_B = self.G(fake_A)
                loss_cycle_B = self.criterion_cycle(recov_B, real_B)

                # 정체성 손실 (선택적)
                loss_identity_A = self.criterion_identity(self.F(real_A), real_A)
                loss_identity_B = self.criterion_identity(self.G(real_B), real_B)

                # 총 생성기 손실
                loss_G = (loss_GAN_AB + loss_GAN_BA) + lambda_cyc * (loss_cycle_A + loss_cycle_B) + lambda_id * (
                            loss_identity_A + loss_identity_B)
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

                # 로그 출력
                print(f"[Epoch {epoch}/{self.epochs}] [Batch {i}/{len(dataloader_A)}] "
                      f"[D_A loss: {loss_D_A.item()}] [D_B loss: {loss_D_B.item()}] "
                      f"[G loss: {loss_G.item()}]")

            # 학습률 스케줄러 단계 업데이트
            scheduler_G.step()
            scheduler_D_A.step()
            scheduler_D_B.step()

            # checkpoint 저장
            save_interval = 5
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(epoch + 1, self.G, self.F, self.D_A, self.D_B,
                                self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B, checkpoint_dir)
