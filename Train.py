import argparse
import torchvision
import torch
from torch.utils import data


def train(EPOCH, dataloader, test_dataloader, optimizer, loss_function, network, model_name, negative_loss, device, should_view, f):
    print('Saving logs...')
    loss_sum = 0
    test_count = 0
    for _, (b_x, _) in enumerate(test_dataloader):
        b_x = b_x.to(device)
        if should_view:
            formatted_b_x = b_x.view(b_x.shape[0], -1)
        else:
            formatted_b_x = b_x
        output, _ = network(formatted_b_x)
        output = output.view(b_x.shape)
        loss = loss_function(output, b_x)
        if negative_loss:
            loss = - loss
        loss_sum += loss
        test_count += 1
    f.write("{} {}\n".format(0, loss_sum / test_count))
    for epoch in range(EPOCH):
        for step, (b_x, _) in enumerate(dataloader):
            b_x = b_x.to(device)
            if should_view:
                formatted_b_x = b_x.view(b_x.shape[0], -1)
            else:
                formatted_b_x = b_x
            output, _ = network(formatted_b_x)
            output = output.view(b_x.shape)
            loss = loss_function(output, b_x)
            if negative_loss:
                loss = - loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Epoch: ', epoch + 1, '| Step: ', step + 1, '| Train loss: %.4f' % loss.cpu().data.numpy())
            if step % SAVE_STEP == 1:
                print('Saving models...')
                torch.save(network, model_name)
        print('Saving models...')
        torch.save(network, model_name)
        print('Saving logs...')
        loss_sum = 0
        test_count = 0
        for _, (b_x, _) in enumerate(test_dataloader):
            b_x = b_x.to(device)
            if should_view:
                formatted_b_x = b_x.view(b_x.shape[0], -1)
            else:
                formatted_b_x = b_x
            output, _ = network(formatted_b_x)
            output = output.view(b_x.shape)
            loss = loss_function(output, b_x)
            if negative_loss:
                loss = - loss
            loss_sum += loss
            test_count += 1
        f.write("{} {}\n".format(epoch + 1, loss_sum / test_count))


parser = argparse.ArgumentParser(
    description = 'Training utility for paper "Investigation on different loss function in image autoencoder"')
parser.add_argument('--save_at', type = int, dest = "SAVE_STEP", help = "Save network at how many steps (default: 10)  "
                                                                        ",", default = 10)
parser.add_argument('--epoch', '-e', type = int, dest = "EPOCH", help = "Epoch for training", default = 10)
parser.add_argument('--batch_size', '-b', type = int, dest = "BATCH_SIZE", help = "Batch size for dataloader",
                    default = 1024)
parser.add_argument('--learning_rate', '-l', type = float, dest = "LR", help = "Learning rate", default = 0.001)
parser.add_argument('--dataset', type = str, dest = "dataset", default = "cifar10", choices = ['cifar10', 'mnist'])
parser.add_argument('--network', type = str, dest = "network", default = "mlp", choices = ['mlp', 'conv'])
parser.add_argument('origional_size', type = int, help = "Size of origional image. Please note that for ConvNet, it is the number of images' channel.")
parser.add_argument('bottleneck', type = int, help = "Size of bottle neck. Please note that for ConvNet, its \"bottleneck\" is input * origional size in one channel / 4")
parser.add_argument('--loss_func', type = str, dest = "loss_function", default = "mse",
                    choices = ['mse', 'l1', 'ssim', 'psnr', 'l1s'])
parser.add_argument('--model_name', type = str, dest = "model_name", help = "Model name for saving & restoring model",
                    default = "saved.pkl")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--log', type = str, default = "train.log",dest = "log_file", help = "Plase to store logs")
args = parser.parse_args()
SAVE_STEP = args.SAVE_STEP
EPOCH = args.EPOCH
BATCH_SIZE = args.BATCH_SIZE
LR = args.LR
dataset = args.dataset
network = args.network
orgsize = args.origional_size
bottleneck = args.bottleneck
loss_function = args.loss_function
model_name = args.model_name
log_file = args.log_file
f = open(log_file, "a")
f.write("x y\n")
if dataset == "cifar10":
    train_data = torchvision.datasets.CIFAR10(
        root = './cifar10/',
        transform = torchvision.transforms.ToTensor(),
        download = True,
    )
    test_data = torchvision.datasets.CIFAR10(
        root = './cifar10/',
        transform = torchvision.transforms.ToTensor(),
        download = True,
        train = False
    )
else:
    train_data = torchvision.datasets.MNIST(
        root = './mnist/',
        transform = torchvision.transforms.ToTensor(),
        download = True,
    )
    test_data = torchvision.datasets.MNIST(
        root = './mnist/',
        transform = torchvision.transforms.ToTensor(),
        download = True,
        train = False
    )
if network == "mlp":
    import mlp_network
    should_view = True
    network = mlp_network.autoencoder(orgsize, bottleneck)
else:
    import conv_network
    should_view = False
    network = conv_network.autoencoder(orgsize, bottleneck)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print("Using CUDA...")
    network.cuda()
else:
    args.device = torch.device('cpu')
negative_loss = False
if loss_function == "mse":
    loss_func = torch.nn.MSELoss()
elif loss_function == "l1":
    loss_func = torch.nn.L1Loss()
elif loss_function == "l1s":
    loss_func = torch.nn.SmoothL1Loss()
elif loss_function == "ssim":
    import pytorch_ssim
    loss_func = pytorch_ssim.SSIM()
    negative_loss = True
else:
    import psnr
    loss_func = psnr.PSNR()
train_loader = data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = data.DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = False)
optimizer = torch.optim.Adam(network.parameters(), lr = LR)
train(EPOCH, train_loader, test_loader, optimizer, loss_func, network, model_name, negative_loss, args.device, should_view, f)
f.close()
torch.cuda.empty_cache()
