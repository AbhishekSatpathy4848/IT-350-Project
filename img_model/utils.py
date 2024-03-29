import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    
class EncDecLoss(nn.Module):
    def __init__(self):
        super(EncDecLoss, self).__init__()
        self.a = 1
        self.b = 0.75

    def mean_square(self, A, B):
        return torch.mean(torch.mul(A - B, A - B), dim=(0, 1, 2, 3))

    def forward(self, cover, secret, stego, stego_decoded):
        return self.a * self.mean_square(cover, stego) + self.b * self.mean_square(secret, stego_decoded)
    
import tqdm

def train_discriminator(num_epochs, img_dataloader, enc, disc, criterion, discOptim, device):
    enc.eval()
    disc.train()
    for epoch in range(num_epochs):
        avg_loss = 0
        with tqdm.tqdm(img_dataloader, unit="batch") as pbar:
            for i, data in enumerate(pbar, 0):
                discOptim.zero_grad()
                #data is a tensor of size (batch_size, 6, 256, 256)
                data = data.to(device)
                cover = data[:, :3, :, :]
                stego = enc(data)
                real = torch.zeros((cover.shape[0], 1), device=device)
                fake = torch.ones((cover.shape[0], 1), device=device)
                real_loss = criterion(disc(cover), real)
                fake_loss = criterion(disc(stego), fake)
                loss = real_loss + fake_loss
                loss.backward()
                discOptim.step()

                avg_loss += loss.item()
                pbar.set_postfix({"Avg. Loss (Discriminator)": avg_loss / (i + 1)})
                pbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")

def train(num_epochs, img_dataloader, enc, dec, encDecCriterion, encDecOptim, disc, criterion, discOptim, device, mix_coeff):
    for epoch in range(num_epochs):
        with tqdm.tqdm(img_dataloader, unit="batch") as pbar:
            avg_gen_loss = 0
            avg_disc_loss = 0
            gen_iter = 0
            disc_iter = 0
            for i, data in enumerate(pbar, 0):
                if i % 2 == 0:
                    enc.train()
                    dec.train()
                    disc.eval()
                    #update generator
                    encDecOptim.zero_grad()
                    data = data.to(device)
                    cover = data[:, :3, :, :]
                    secret = data[:, 3:, :, :]
                    stego = enc(data)
                    stego_decoded = dec(stego)
                    gen_loss = criterion(disc(stego), torch.zeros((cover.shape[0], 1), device=device))
                    gen_loss_full = encDecCriterion(cover, secret, stego, stego_decoded) + mix_coeff * gen_loss
                    gen_loss_full.backward()
                    encDecOptim.step()

                    avg_gen_loss += gen_loss_full.item()
                    gen_iter += 1

                enc.eval()
                dec.eval()
                disc.train()
                #update discriminator
                discOptim.zero_grad()
                data = data.to(device)
                cover = data[:, :3, :, :]
                secret = data[:, 3:, :, :]
                stego = enc(data)
                real = torch.zeros((cover.shape[0], 1), device=device)
                fake = torch.ones((cover.shape[0], 1), device=device)
                real_loss = criterion(disc(cover), real)
                fake_loss = criterion(disc(stego), fake)
                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                discOptim.step()

                avg_disc_loss += disc_loss.item()
                disc_iter += 1
                pbar.set_postfix({"Avg. Loss (Discriminator)": avg_disc_loss / disc_iter, "Avg. Loss (Generator)": avg_gen_loss / gen_iter})
                pbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")

def infer_random(dataset, enc, dec, device):
    enc.eval()
    dec.eval()
    idx = random.randint(0, len(dataset) - 1)
    data = dataset[idx]
    data = data.unsqueeze(0).to(device)
    stego = enc(data)
    stego_decoded = dec(stego)
    stego = stego.to("cpu").detach()
    stego_decoded = stego_decoded.to("cpu").detach()
    cover = data[:, :3, :, :].to("cpu").detach()
    secret = data[:, 3:, :, :].to("cpu").detach()
    return cover, secret, stego, stego_decoded

def view_random_inference(dataset, enc, dec, device):
    cover, secret, stego, stego_decoded = infer_random(dataset, enc, dec, device)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(cover[0].permute(1, 2, 0))
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Cover Image")
    ax[0, 1].imshow(secret[0].permute(1, 2, 0))
    ax[0, 1].axis("off")
    ax[0, 1].set_title("Secret Image")
    ax[1, 0].imshow(stego[0].permute(1, 2, 0))
    ax[1, 0].axis("off")
    ax[1, 0].set_title("Stego Image")
    ax[1, 1].imshow(stego_decoded[0].permute(1, 2, 0))
    ax[1, 1].axis("off")
    ax[1, 1].set_title("Decoded Stego Image")
    plt.show()

def get_random_images(dataset):
    idx = random.randint(0, len(dataset) - 1)
    data = dataset[idx]
    cover = data[:3, :, :]
    secret = data[3:, :, :]
    cover = cover.permute(1, 2, 0)
    secret = secret.permute(1, 2, 0)
    return cover, secret

def view_random_images(dataset, num_rows):
    plt.figure(figsize=(6, num_rows * 3))
    for i in range(num_rows):
        cover, secret = get_random_images(dataset)
        plt.subplot(num_rows, 2, 2 * i + 1)
        plt.imshow(cover)
        plt.axis("off")
        plt.subplot(num_rows, 2, 2 * i + 2)
        plt.imshow(secret)
        plt.axis("off")
    plt.show()

def save_models(enc, dec, disc):
    torch.save(enc.state_dict(), "enc.pt")
    torch.save(dec.state_dict(), "dec.pt")
    torch.save(disc.state_dict(), "disc.pt")

def infer_on(enc, dec, cover, secret, device):
    enc.eval()
    dec.eval()
    data = torch.cat((cover, secret), dim=0).unsqueeze(0).to(device)
    stego = enc(data)
    stego_decoded = dec(stego)
    stego = stego.to("cpu").detach()
    stego_decoded = stego_decoded.to("cpu").detach()
    cover = data[:, :3, :, :].to("cpu").detach()
    secret = data[:, 3:, :, :].to("cpu").detach()
    return stego, stego_decoded