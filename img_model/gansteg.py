from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import torchvision.transforms.v2 as transforms
from models import StegEncoder, StegDecoder, Discriminator
from dataset import StegDataset
import utils

do_training = False

workers = 1
batch_size = 8
num_epochs_disc = 2
num_epochs_together = 15
image_size = 256
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir = "sample-train-imgs/"

enc = StegEncoder()
dec = StegDecoder()
disc = Discriminator()
enc = torch.nn.DataParallel(enc).to(device)
dec = torch.nn.DataParallel(dec).to(device)
disc = torch.nn.DataParallel(disc).to(device)

if do_training:
    train_images = [Image.open(img_dir + img) for img in os.listdir(img_dir)]
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[0:3, :, :] if x.shape[0] == 4 else x)
    ])
    train_dataset = StegDataset(train_images, transform_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    encDecOptim = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, betas=(beta1, 0.999))
    discOptim = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

    utils.weights_init(enc)
    utils.weights_init(dec)
    utils.weights_init(disc)

    discCriterion = torch.nn.BCELoss()
    encDecCriterion = utils.EncDecLoss()

    utils.view_random_images(train_dataset, 4)
    utils.train_discriminator(num_epochs_disc, train_dataloader, enc, disc, discCriterion, discOptim, device)
    utils.view_random_images(train_dataset, 4)
    utils.train(num_epochs_together, train_dataloader, enc, dec, encDecCriterion, encDecOptim, disc, discCriterion, discOptim, device, 1)
    utils.view_random_images(train_dataset, 4)
    utils.save_models(enc, dec, disc)
else:
    enc.load_state_dict(torch.load("enc.pt"))
    dec.load_state_dict(torch.load("dec.pt"))
    disc.load_state_dict(torch.load("disc.pt"))

transform_test = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[0:3, :, :] if x.shape[0] == 4 else x)
])
# test_images = [Image.open(img_dir + img) for img in os.listdir(img_dir)]
# test_dataset = StegDataset(test_images, transform_test)

# utils.view_random_inference(test_dataset, enc, dec, device)
    
def encode_decode(cover, secret):
    cover = transform_test(cover)
    secret = transform_test(secret)
    stego, stego_decoded = utils.infer_on(enc, dec, cover, secret, device)
    #convert to PIL images
    stego = transforms.ToPILImage()(stego)
    stego_decoded = transforms.ToPILImage()(stego_decoded)
    return stego, stego_decoded