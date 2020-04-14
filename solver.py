import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


import model
import utilities


def get_loader(dataset, opts):

	transform = transforms.Compose([
		transforms.Scale(opts.image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))])

	if dataset == 'USPS' :
		usps_train = datasets.USPS(root = '/content/CycleGan/data/USPS',train=True, download=True, transform = transform)
		usps_test = datasets.USPS(root = '/content/CycleGan/data/USPS',train=False, download=True, transform = transform)
		train_dloader = DataLoader(dataset=usps_train,
									batch_size=opts.batch_size,
									shuffle=True,
									num_workers=opts.num_workers)

		test_dloader = DataLoader(dataset=usps_test,
									train=False,
									batch_size=opts.batch_size,
									shuffle=True,
									num_workers=opts.num_workers)

	if dataset == 'MNIST':
		mnis_train = datasets.MNIST(root='/content/CycleGan/data/MNIST',train=True, download=True, transform=transform)
		mnis_test = datasets.MNIST(root='/content/CycleGan/data/MNIST',train=False, download=True, transform=transform)
		train_dloader = DataLoader(dataset=mnis_train,
									Train = True,
									batch_size=opts.batch_size,
									shuffle=True,
									num_workers=opts.num_workers)

		test_dloader = DataLoader(dataset=mnis_test,
									Train = False,
									batch_size=opts.batch_size,
									shuffle=True,
									num_workers=opts.num_workers)

	return train_dloader, test_dloader

def prints_models(G_XtoY, G_YtoX, D_X, D_Y):
	'''
	print model information
	'''
	print("						G_XtoY					")
	print("---------------------------------------------")
	print(G_XtoY)
	print("---------------------------------------------")

	print("						G_YtoX					")
	print("---------------------------------------------")
	print(G_YtoX)
	print("---------------------------------------------")

	print("						D_X						")
	print("---------------------------------------------")
	print(D_X)
	print("---------------------------------------------")

	print("						D_Y						")
	print("---------------------------------------------")
	print(D_Y)
	print("---------------------------------------------")

def create_model(opts):
    """Builds the generators and discriminators.
    """
    
    ### CycleGAN
    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.cuda.is_available():
    	G_XtoY.cuda()
    	G_YtoX.cuda()
    	D_X.cuda()
    	D_Y.cuda()
    	print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y

def train(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_loader(dataset=opts.X, opts=opts)
    dataloader_Y, test_dataloader_Y = get_loader(dataset=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)

    # Start training
    G_XtoY, G_YtoX, D_X, D_Y = cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts)

    return G_XtoY, G_YtoX, D_X, D_Y

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
    	if opts.__dict__[key]:
    		print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

def cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = to_var(test_iter_X.next()[0])
    fixed_Y = to_var(test_iter_Y.next()[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    try:
    	for iteration in range(1, opts.train_iters+1):

    		# Reset data_iter for each epoch
    		if iteration % iter_per_epoch == 0:
    			iter_X = iter(dataloader_X)
    			iter_Y = iter(dataloader_Y)

    		images_X, labels_X = iter_X.next()
    		images_X, labels_X = to_var(images_X), to_var(labels_X).long().squeeze()

    		images_Y, labels_Y = iter_Y.next()
    		images_Y, labels_Y = to_var(images_Y), to_var(labels_Y).long().squeeze()

          # ============================================
          #            TRAIN THE DISCRIMINATORS
          # ============================================

          # Train with real images

    		d_optimizer.zero_grad()
    		outX = D_X(images_X)
    		outY = D_Y(images_Y)

          # 1. Compute the discriminator losses on real images
    		D_X_loss = torch.mean((outX - labels_X)**2)
    		D_Y_loss = torch.mean((outY - labels_Y)**2)

    		d_real_loss = D_X_loss + D_Y_loss
    		d_real_loss.backward()
    		d_optimizer.step()

          # Train with fake images
    		d_optimizer.zero_grad()

		 # 2. Generate fake images that look like domain X based on real images in domain Y
    		fake_X = G_YtoX(images_Y)
    		out = D_X(fake_X)

          # 3. Compute the loss for D_X
    		D_X_loss = torch.mean(out**2)

          # 4. Generate fake images that look like domain Y based on real images in domain X

    		fake_Y = G_XtoY(images_X)
    		out = D_Y(fake_Y)

          # 3. Compute the loss for D_Y

    		D_Y_loss = torch.mean(out**2)

    		d_fake_loss = D_X_loss + D_Y_loss
    		d_fake_loss.backward()
    		d_optimizer.step()

          # =========================================
          #            TRAIN THE GENERATORS
          # =========================================

    		g_optimizer.zero_grad()

          # 1. Generate fake images that look like domain X based on real images in domain Y

    		fake_X = G_YtoX(images_Y)
    		out = D_X(fake_X)

          # 2. Compute the generator loss based on domain X
			#g_loss = torch.mean((out- labels_X)**2)
    		g_loss = torch.mean((out - 1)**2)

    		reconstructed_Y = G_XtoY(fake_X)

          # 3. Compute the cycle consistency loss (the reconstruction loss)
    		cycle_consistency_loss = torch.mean((images_Y- reconstructed_Y)**2)

    		g_loss += opts.lambda_cycle * cycle_consistency_loss

    		g_loss.backward()
    		g_optimizer.step()

    		g_optimizer.zero_grad()

          # 1. Generate fake images that look like domain Y based on real images in domain X
    		fake_Y = G_XtoY(images_X)
    		out = D_Y(fake_Y)

          # 2. Compute the generator loss based on domain Y
			#g_loss = torch.mean((out - labels_Y)**2)
    		g_loss = torch.mean((out - 1)**2)

    		reconstructed_X = G_YtoX(fake_Y)
          # 3. Compute the cycle consistency loss (the reconstruction loss)

    		cycle_consistency_loss = torch.mean((images_X- reconstructed_X)**2)

    		g_loss += opts.lambda_cycle * cycle_consistency_loss

    		g_loss.backward()
    		g_optimizer.step()

          # Print the log info

    		if iteration % opts.log_step == 0:
    			print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
    				'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
    					iteration, opts.train_iters, d_real_loss.item(), D_Y_loss.item(),
    					D_X_loss.item(), d_fake_loss.item(), g_loss.item()))


          # Save the generated samples
    		if iteration % opts.sample_every == 0:
    			cyclegan_save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

          # Save the model parameters
    		if iteration % opts.checkpoint_every == 0:
    			cyclegan_checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)

    except KeyboardInterrupt:
    	print('Exiting early from training.')
    	return G_XtoY, G_YtoX, D_X, D_Y

    return G_XtoY, G_YtoX, D_X, D_Y
