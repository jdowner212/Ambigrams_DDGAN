import sys
import os
from os.path import dirname, abspath, join, exists

PARENT_DIR  = dirname(abspath(__file__))
MODELS_DIR  = os.path.join(PARENT_DIR,'models')

if PARENT_DIR not in sys.path:
    sys.path = [PARENT_DIR] + sys.path

from my_DDGAN import *
from general_utils import *
from file_utils import *


def init_weights(m):
    if type(m) == ConvTranspose2d:
        torch.nn.init.normal_(m.weight, 0.0, 0.02).cuda()
    elif type(m) == BatchNorm2d:
        torch.nn.init.normal_(m.weight, 1.0, 0.02).cuda()
        torch.nn.init.constant_(m.bias, 0).cuda()


def save_k_images(img_batch, next_model_name, epoch, k=25, model_folder=MODELS_DIR):
    img_batch_rot = rot180(torch.tensor(img_batch)).detach().clone().numpy()
    assert k < len(img_batch)
    assert os.path.exists(f'{model_folder}/{next_model_name}/epoch_{epoch}')

    try_to_make_folder(f'{model_folder}/{next_model_name}/epoch_{epoch}/images', print_message='\nFailed to create image folder -- may already exist\n(function: save_k_images)')

    letters = next_model_name[:2]
    for i in range(k):
        img     = prep_img_array_for_plot(img_batch[i])
        img_rot = prep_img_array_for_plot(img_batch_rot[i])

        fig, axs = plt.subplots(1,2)
        axs[0].imshow(img)
        axs[0].set_title(letters[0])
        axs[0].axis('off')
        axs[1].imshow(img_rot)
        axs[1].set_title(letters[1])
        axs[1].axis('off')

        fig.savefig(f'{model_folder}/{next_model_name}/epoch_{epoch}/images/{i}.png')
        plt.close()

def plot_results(epoch, imgs, lossD1_reals, lossD2_reals, lossD1_fakes, lossD2_fakes, lossGs, next_model_name, batch_count, y_max=20, save=True, model_folder=MODELS_DIR):
    gray = True if imgs.shape[1] == 1 else False
    fig = plt.figure(figsize=(14, 6))
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.0, width_ratios= [1, 1.3333])


    for i in range(2):

        if i % 2 == 0:
            inner_grid = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=outer_grid[i], wspace=0.05, hspace=0.01)
            a, b = int(i/4)+1,i%4+1

            for j, (c, d) in enumerate(product(range(1, 6), repeat=2)):
                ax = plt.Subplot(fig, inner_grid[j])
                this_image = prep_img_array_for_plot(imgs[j],gray)
                ax.imshow(this_image)
                ax.axis('off')
                fig.add_subplot(ax)
        else:
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
            ax = plt.Subplot(fig, inner_grid[0])
            ax.set_ylim([0, y_max])
            for pair in [(lossD1_reals, 'D1_real'), (lossD2_reals, 'D2_real'), (lossD1_fakes,'D1_fake'),(lossD2_fakes,'D2_fake'),(lossGs, 'gen')]:
                ax.plot(range(len(pair[0])),pair[0], label=f'{pair[1]} loss')
            ax.legend()
            ax.set_xlabel(f'Batches processed ({batch_count} per epoch)')
            ax.set_ylabel('Loss')
            ax.set_title('Loss per batch')

            fig.add_subplot(ax)
            if save==True:
                try_to_make_folder(f'{model_folder}/{next_model_name}/epoch_{epoch}')
                fig.savefig(f'{model_folder}/{next_model_name}/epoch_{epoch}/graph.png')

    all_axes = fig.get_axes()

    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)

    plt.show()

def new_or_existing_models(G_path, D1_path, D2_path, node_factor, channels, image_size, activation_G, activation_D, letters, model_folder = MODELS_DIR):
    starting_epoch = 0
    if G_path != None and D1_path != None and D2_path != None:
        netG  = torch.load(G_path)
        netD1 = torch.load(D1_path)
        netD2 = torch.load(D2_path)

        assert model_name_from_path(G_path) == model_name_from_path(D1_path) and model_name_from_path(G_path) == model_name_from_path(D2_path)
        model_name = model_name_from_path(G_path)
        starting_epoch = current_epoch(G_path)

        if starting_epoch < largest_epoch(G_path):
            model_name = model_name + f'_branch_epoch_{starting_epoch}'
            create_model_folder(letters, model_name)

    else:
        netG = Generator(node_factor, channels, image_size, activation = activation_G).to('cuda')
        netG.apply(init_weights)
        netD1 = Discriminator(node_factor,channels, activation = activation_D).to('cuda')
        netD1.apply(init_weights)
        netD2 = Discriminator(node_factor,channels, activation = activation_D).to('cuda')
        netD2.apply(init_weights)
        
        model_name = get_next_model_name(letters, model_folder)
        create_model_folder(letters, model_name)
    
    return netG, netD1, netD2, model_name, starting_epoch


def save_model(letters, epoch, batch_size, image_size, netD1, netD2, netG, opt_D1, opt_D2, opt_G, next_model_name, model_folder=MODELS_DIR):
    
    assert netD1.node_factor == netD2.node_factor and netD1.node_factor == netG.node_factor and netD1.activation == netD2.activation
    assert list(opt_D1.param_groups[0].values())[1:] == list(opt_D2.param_groups[0].values())[1:]
    
    act_D, act_G = netD1.activation, netG.activation
    node_factor  = netD1.node_factor
    lr_D, betas_D, eps_D, weights_dec_D, _, _, _, _ = list(opt_D2.param_groups[0].values())[1:]
    lr_G, betas_G, eps_G, weights_dec_G, _, _, _, _  = list(opt_G.param_groups[0].values())[1:]

    with open(f'{model_folder}/{next_model_name}/log.txt', 'w') as f:
        as_string = model_as_string(next_model_name, lr_D, lr_G, weights_dec_D, weights_dec_G, betas_D, betas_G, act_D, act_G, node_factor, batch_size)
        f.write(as_string)                                   
    f.close()

    try_to_make_folder(f'{model_folder}/{next_model_name}/epoch_{epoch}/models', print_message = f"\nIssue creating {model_folder}/{next_model_name}/{epoch}/models -- may already exist\n(function: save_model)")
    for (net,net_name) in [(netD1, 'netD1'), (netD2, 'netD2'), (netG, 'netG')]:
        torch.save(net, f'{model_folder}/{next_model_name}/epoch_{epoch}/models/{net_name}_epoch_{epoch}.pt')


def train_multiple_GAN(L1_batches, 
                       L2_batches, 
                       letters, 
                       lr_D,
                       lr_G,
                       weights_dec_D,
                       weights_dec_G,
                       betas_D,
                       betas_G,
                       activation_D,
                       activation_G,
                       node_factor,
                       epochs,
                       model_folder = MODELS_DIR, 
                       D1_path = None,
                       D2_path = None,
                       G_path  = None,
                       #current_epoch = 0
                       save_plots=True, 
                       plot_every=1,
                       test_compile=False):

    assert len(L1_batches[0].shape) == len(L2_batches[0].shape)

    if test_compile == True:
        L1_batches = {i: L1_batches[i] for i in range(5)}
        L2_batches = {i: L2_batches[i] for i in range(5)}
        epochs=1


    batch_count = len(L1_batches)
    batch_size  = len(L1_batches[0])

    image_size  = L1_batches[0][0].shape[1]
    channels    = L1_batches[0][0].shape[0]


    # Continues training if existing models provided, otherwise creates new ones
    netG, netD1, netD2, next_model_name, starting_epoch = new_or_existing_models(G_path, D1_path, D2_path, node_factor, channels, image_size, activation_G, activation_D, letters, model_folder)
    if G_path != None:
        lr_D, lr_G, weights_dec_D, weights_dec_G, betas_D, betas_G, netD_activation, netG_activation, node_factor = parse_parameteres_from_log(G_path)

    ## create optimizers
    opt_D1 = optim.Adam(netD1.parameters(), lr=lr_D, betas=betas_D, weight_decay=weights_dec_D)
    opt_D2 = optim.Adam(netD2.parameters(), lr=lr_D, betas=betas_D, weight_decay=weights_dec_D)
    opt_G  = optim.Adam(netG.parameters(),  lr=lr_G, betas=betas_G, weight_decay=weights_dec_G)


    as_string = model_as_string(next_model_name, lr_D, lr_G, weights_dec_D, weights_dec_G, betas_D, betas_G, activation_D, activation_G, node_factor, batch_size)
    print(as_string)

    lossD1_reals, lossD2_reals, lossD1_fakes, lossD2_fakes, lossGs = [], [], [], [], []
    print('epochs + starting_epoch:',epochs+starting_epoch)
    for e in range(starting_epoch, epochs+starting_epoch):
        try_to_make_folder(os.path.join(model_folder, next_model_name, f'epoch_{e}'))
        print('Epoch', e)
        for i in tqdm(range(batch_count)):
            sys.stdout.write('\r'+ f'Batch {i}/{batch_count}')
            time.sleep(0.5)

            L1_batch = L1_batches[i]
            L2_batch = rot180(L2_batches[i]) 

            updates = update_my_DDGAN(L1_batch, L2_batch, netD1, netD2, netG, opt_D1, opt_D2, opt_G, 'cuda')
            netD1,  netD2,  netG                                        = updates[0]
            opt_D1, opt_D2, opt_G                                       = updates[1]
            lossD1_real, lossD2_real, lossD1_fake, lossD2_fake, lossG   = updates[2]
            fake_img                                                    = updates[3].clone().detach()

            lossD1_reals.append(lossD1_real.item())
            lossD2_reals.append(lossD2_real.item())
            lossD1_fakes.append(lossD1_fake.item())
            lossD2_fakes.append(lossD2_fake.item())
            lossGs.append(lossG.item())

        sys.stdout.write('\r' + f'Epoch {e%(starting_epoch+epochs)}/{(starting_epoch+epochs)}')
        time.sleep(0.5)

        to_plot = fake_img.cpu().numpy()
        save_k_images(to_plot, next_model_name, e)
        save_model(letters, e, batch_size, image_size, netD1, netD2, netG, opt_D1, opt_D2, opt_G, model_folder=model_folder, next_model_name = next_model_name)

        if e%plot_every == 0:
            plot_results(e, to_plot, lossD1_reals, lossD2_reals, lossD1_fakes, lossD2_fakes, lossGs, next_model_name, batch_count = batch_count, save=save_plots)
            print(f"\nGenerator loss: {round(lossG.item(),4)}")
            print(f"D1_real loss: {round(lossD1_real.item(),4)}")
            print(f"D2_real loss: {round(lossD2_real.item(),4)}")
            print(f"D1_fake loss: {round(lossD1_fake.item(),4)}")
            print(f"D2_fake loss: {round(lossD2_fake.item(),4)}")

def generate_sample(model_name, epoch, count=10, model_folder=MODELS_DIR):
    complete_path = os.path.join(model_folder, f'{model_name}/epoch_{epoch}/models/netG_epoch_{epoch}.pt')
    print('model_folder:',model_folder)
    print('path: ', complete_path)
    this_net_G = torch.load(complete_path)
    noise = torch.randn(count,100,1,1,device='cuda')
    fake_image = this_net_G(noise)
    return fake_image
