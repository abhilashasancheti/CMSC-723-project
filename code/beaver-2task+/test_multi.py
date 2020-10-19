# # nn.DataParallel implementeation woorking fine

# import os
# from datetime import datetime
# import argparse
# import torch.multiprocessing as mp
# import torchvision
# import torchvision.transforms as transforms
# import torch
# import torch.nn as nn
# import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


# def cleanup():
#     dist.destroy_process_group()
    
# def train(gpu, args):
# 	model = ConvNet()
# 	setup(0, 2)
# 	# setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
# 	# rank 2 uses GPUs [4, 5, 6, 7].
# 	n = torch.cuda.device_count() // world_size
# 	device_ids = list(range(rank * n, (rank + 1) * n))
# 	# output_device defaults to device_ids[0]
# 	#model = nn.DataParallel(model, device_ids=[0])
# 	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
# # 	torch.cuda.set_device(gpu)
# # 	model.cuda(gpu)
# 	batch_size = 100
# 	# define loss function (criterion) and optimizer
# 	criterion = nn.CrossEntropyLoss().cuda(gpu)
# 	optimizer = torch.optim.SGD(model.parameters(), 1e-4)
# 	# Data loading code
# 	train_dataset = torchvision.datasets.MNIST(root='./',
# 												train=True,
# 												transform=transforms.ToTensor(),
# 												download=True)
# 	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
# 											batch_size=batch_size,
# 											shuffle=True,
# 											num_workers=0,
# 											pin_memory=True)

# 	start = datetime.now()
# 	total_step = len(train_loader)
# 	for epoch in range(args.epochs):
# 		for i, (images, labels) in enumerate(train_loader):
# 			images = images.cuda(non_blocking=True)
# 			labels = labels.cuda(non_blocking=True)
# 			# Forward pass
# 			outputs = model(images)
# 			print("Outside: input size", images.shape,"output_size", outputs.shape)
# 			loss = criterion(outputs, labels)

# 			# Backward and optimize
# 			optimizer.zero_grad()
# 			loss.backward()
# 			optimizer.step()
# 			if (i + 1) % 100 == 0 and gpu == 0:
# 				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
# 					epoch + 1, 
# 					args.epochs, 
# 					i + 1, 
# 					total_step,
# 					loss.item())
# 					)
# 	cleanup()
# 	if gpu == 0:
# 		print("Training complete in: " + str(datetime.now() - start))
            


# class ConvNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(7*7*32, num_classes)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         print("\tIn Model: input size", x.shape,"output size", out.shape)
#         return out


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--nodes', default=1,
#                         type=int, metavar='N')
#     parser.add_argument('-g', '--gpus', default=1, type=int,
#                         help='number of gpus per node')
#     parser.add_argument('-nr', '--nr', default=0, type=int,
#                         help='ranking within the nodes')
#     parser.add_argument('--epochs', default=2, type=int, 
#                         metavar='N',
#                         help='number of total epochs to run')
#     args = parser.parse_args()
#     train(0,args)


# main()

import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP




class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    #n = torch.cuda.device_count() // world_size
    device_ids = list(range(2,4))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=[2,3], output_device=2)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = torch.device(dev0)
        self.dev1 = torch.device(dev1)
        self.net1 = torch.nn.Linear(10, 10).to(torch.device(dev0))
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(torch.device(dev1))

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()

def demo_model_parallel(rank, world_size):
    setup(rank, world_size)

    # setup mp_model and devices for this process
    print("rank", rank)
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(torch.device(dev0), torch.device(dev1))
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(torch.device(dev1))
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    demo_basic(0, 2)
#     run_demo(demo_checkpoint, 1)
#     print("running parallel now ..")
#     if torch.cuda.device_count() >= 2:
#         run_demo(demo_model_parallel, 1)
