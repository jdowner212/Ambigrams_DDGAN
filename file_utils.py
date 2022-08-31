import sys
import os
from os.path import dirname, abspath, join, exists

PARENT_DIR  = dirname(abspath(__file__))
if PARENT_DIR not in sys.path:
    sys.path = [PARENT_DIR] + sys.path

from general_utils import *

def parse_parameteres_from_log(path):
    with open (log_path_from_model_path(path),'r') as f:
        lines = f.readlines()
        lr_D = float(lines[3][19:].split(',')[0].split(' = ')[1])
        lr_G = float(lines[4][19:].split(',')[0].split(' = ')[1])
        weights_dec_D = float(lines[5][19:].split(',')[0].split(' = ')[1]) 
        weights_dec_G = float(lines[6][19:].split(',')[0].split(' = ')[1]) 
        b1_D = float((lines[7][19:].split(','))[0].split('(')[1])
        b2_D = float((lines[7][19:].split(','))[1].split(')')[0])
        b1_G = float((lines[8][19:].split(','))[0].split('(')[1])
        b2_G = float((lines[8][19:].split(','))[1].split(')')[0])
        betas_D = (b1_D, b2_D)
        betas_G = (b1_G, b2_G)
        netD_activation = Sigmoid
        netG_activation = Sigmoid if 'Sigmoid' in lines[10] else Tanh
        node_factor = int(lines[11][19:].split(' = ')[1].split(',')[0])
    f.close()
    return lr_D, lr_G, weights_dec_D, weights_dec_G, betas_D, betas_G, netD_activation, netG_activation, node_factor


def create_base_images(letter, img_size=240, data_dir=DATA_DIR, fonts_folder=FONTS_DIR):
    '''
    Collect image of given letter in both upper and lower case in every font.

    This forms the 'base' folder for this particular letter -- the folder of
    original images that will later be multiplied and augmented to increase
    the size of the training data.
    '''
    cs = [letter.lower(), letter.upper()]

    # Position of the letter will be somewhat randomized, but establish some
    # spatial intervals to help make sure letter is within outer bounds of the image
    half    = int(img_size/2)
    third   = int(img_size/3)
    fourth  = int(img_size/4)
    sixth   = int(img_size/6)
    twelfth = int(img_size/12)

    letter_dict = {}
    font_names = [f for f in os.listdir(fonts_folder) if f[0] != '.']
    
    this_letter_folder = join(data_dir,letter)
    base_folder   = f'{this_letter_folder}/base_{letter}'
    try_to_make_folder(this_letter_folder, print_message=None)
    try_to_make_folder(base_folder,   print_message=None)

    all_images = []
    print(letter)
    for i in range(2):
        letter = cs[i]
        for name in tqdm(font_names):

            # randomized displacement -- had to experiment to find good values
            dX, dY, dFontSize = rc(range(fourth)), rc(range(twelfth)), rc(range(third))
            position_x, position_y = sixth+dX, dY
            if name in ['Signatures.otf', 'JaysenDemoRegular.ttf','Nedilan.otf','Regan Script.otf']: # fonts with unusual sizing
                position_x, position_y = fourth+dX, (5*sixth)+dY

            font_path = join(fonts_folder, name)
            this_font = ImageFont.truetype(font_path, half+dFontSize)

            img=Image.new("RGB", (img_size,img_size),(255,255,255))
            draw = ImageDraw.Draw(img)
            draw.text((position_x, position_y), letter, (0,0,0), font=this_font)
            all_images.append(img)
        
    for i in range(len(all_images)):
        image = all_images[i]
        i = str(i)
        filename = '0'*(4-len(i)) + i + '.png'
        filename = join(base_folder, filename)
        image.save(filename)


def create_transformed_groups(letter, total_number = 30000, data_dir=DATA_DIR, fonts_folder=FONTS_DIR, font_count=FONT_COUNT):
    '''
    Transform letters from 'base' folder and store as 
    '''
    assert font_count == len([f for f in os.listdir(fonts_folder) if f[0] != '.'])
    upper_plus_lower_count = font_count*2

    master_folder = f'{data_dir}/{letter}'
    assert exists(master_folder)
    base_folder = join(master_folder, f'base_{letter}')
    try_to_make_folder(base_folder, print_message=None)

    print(letter)
    print('base folder: ',base_folder)

    file_names = os.listdir(base_folder)
    file_paths_inside = [join(base_folder,f) for f in file_names]
    base_images = [Image.open(f) for f in file_paths_inside]
    number_of_groups = int(np.ceil(total_number/(font_count*2)))

    for g in range(number_of_groups):
        create_new = True
        group_folder = join(data_dir,letter,f'Group_{g}')
        if exists(group_folder):
            create_new = False

            if len(os.listdir(group_folder)) == upper_plus_lower_count:
                print(f'already exists: Group {g}')
            else:
                print(f'incorrect size: Group {g}')
                shutil.rmtree(group_folder)
        
        if create_new == True:
            print('Group',g)

            try_to_make_folder(group_folder, print_message=f'Problem creating folder: {group_folder}\n(function: create_transformed_groups)')

            transformed = [random_Ts(t) for t in tqdm(base_images)]
            random.shuffle(transformed)
            file_paths = [os.path.join(group_folder,f) for f in file_names]
            for t,p in list(zip(transformed, file_paths)):
                t.save(p)

def add_augmented_data(letter, total_number, data_dir=DATA_DIR, img_size = 240, delete_existing=False):
    '''
    Multiply/augment the base folder data -- which you can re-collect by passing 'delete_existing=True'
    '''
    if delete_existing == True:
        this_letter_folder = f'{data_dir}/{letter}'
        print('Removing/re-doing the following folders:')
        for folder in os.listdir(this_letter_folder):
            folder = join(this_letter_folder,folder)
            print(folder)
            shutil.rmtree(folder)
    create_base_images(letter, img_size)
    create_transformed_groups(letter,total_number)

def get_batches_filenames(letter, batch_count, batch_size, data_dir=DATA_DIR):
    '''
    I had difficulty reading data from large folders in Colab, so I split the original data
    among multiple folders, each capped at 1200 images. This required a workaround to
    enable access to a sequential batch of arbitrary size.
    '''    
    get_folder_path      = lambda folder_count: join(data_dir,letter,f'Group_{folder_count}')
    get_filepaths        = lambda folder_path: [join(folder_path,f) for f in os.listdir(folder_path)]
    get_inside_filepaths = lambda folder_count: get_filepaths(get_folder_path(folder_count))
    
    all_batches_filenames = []
    folder_count = 0 
    this_folder_filepaths = get_inside_filepaths(folder_count)
    transformed_folder_length = len(this_folder_filepaths)

    assert batch_size < transformed_folder_length
    spot_in_folder = 0
    for c in range(batch_count):
        this_batch_filenames = []
        while len(this_batch_filenames) < batch_size: 
            if transformed_folder_length - spot_in_folder >= batch_size:
                
                # batch can be drawn from a single folder
                this_batch_filenames += this_folder_filepaths[spot_in_folder:spot_in_folder + batch_size]
                spot_in_folder += batch_size

            else:

                # batch spans two folders
                slice_1 = this_folder_filepaths[spot_in_folder:]

                # move on to first file in next folder
                folder_count += 1  
                spot_in_folder = 0
                this_folder_filepaths = get_inside_filepaths(folder_count)

                slice_2 = this_folder_filepaths[spot_in_folder:(batch_size-len(slice_1))]
                
                this_batch_filenames += (slice_1 + slice_2)
                spot_in_folder += len(slice_2)

        all_batches_filenames.append(this_batch_filenames)

    return all_batches_filenames

def get_batches(letter, total_number_images, image_size=112, batch_size=256):
    batch_count = int(total_number_images/batch_size)
    batches_filenames = get_batches_filenames(letter, batch_count, batch_size)
    batches = {}
    for i in tqdm(range(len(batches_filenames))):
        batches[i] = stack([add_noise(open_as_grayscale_tensor(I,image_size)) for I in batches_filenames[i]])
    return batches

def get_next_model_name(letters, model_folder=MODELS_DIR):
    '''
    Given two letters, return name of next model to be created
    '''
    model_names = [f for f in os.listdir(model_folder) if letters in f]
    model_numbers = []
    possible = True

    i=0

    while possible:
        i-=1
        try:
            more_numbers = sorted([f[i:] for f in model_names if f[i] in '0123456789'], reverse=True )
            model_numbers += more_numbers
            model_numbers=sorted([int(n) for n in model_numbers], reverse=True)
        except:
            possible=False
    next_model_name = f'{letters}_1' if len(model_names) == 0 else f'{letters}_{str(int(model_numbers[0]) + 1)}'   
    return next_model_name


def create_model_folder(letters, next_model_name, model_folder = MODELS_DIR):
    '''
    Create the folder for this specific model and instantiate a 'log.txt' file
    to record training hyperparameters
    '''
    next_model_folder = join(model_folder, next_model_name)
    try_to_make_folder(next_model_folder,print_message='\nran into an issue')

    try:
        log_file = join(next_model_folder, 'log.txt')
        log_file = open(log_file, 'w')
        log_file.close()
    except:
        print('ran into an issue writing log file (function: create_model_folder)')


def prep_img_array_for_plot(img_array, gray=True):
    if gray==True:
        img_array = tensor(img_array).permute(1,2,0)[:,:,0]
    else:
        img_array = np.array(img_array/np.amax(img_array))
        img_array = denorm(tensor(img_array)).cpu().numpy()
        img_array = np.clip(img_array, -1, 1).transpose(1,2,0)
    return img_array



