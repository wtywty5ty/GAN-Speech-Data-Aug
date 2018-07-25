import time
from train_sn_uncon import *
from models.models_uncon import _netG, _netD

phone_list1 = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't']
phone_list2 = ['v', 'w', 'y', 'z', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'ch']
phone_list3 = ['cl', 'dh', 'dx', 'eh', 'el', 'en', 'er', 'ey', 'hh', 'ih', 'ix', 'iy']
phone_list4 = ['jh', 'ng', 'ow', 'oy', 'sh', 'th', 'uh', 'uw', 'zh', 'epi', 'sil', 'vcl']
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'


parser = argparse.ArgumentParser(description='train DCGAN model')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=100, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--map_size', default=[16, 40], help='size of feature map')
parser.add_argument('--phone',  help='phone')
parser.add_argument('--outf', help="path to output files)")
opt = parser.parse_args()

start_time = time.time()
for phone in phone_list1:
    with open(DIR+'/finetune_gpu%d.cfg'%opt.gpu_id, 'a') as f:
        f.write('HNTRAINSGD: TARGETPHONE = %s \n'%phone)

    opt.phone = phone
    opt.outf = 'outf/GAN_array_uncon/%s'%phone
    print(opt)

    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs('%s/checkpoints' % opt.outf, exist_ok=True)


    if torch.cuda.is_available():
        device = torch.device('cuda:%d'%opt.gpu_id)
    else:
        device = torch.device('cpu')
    cudnn.benchmark = True

    CDCGAN_Classifier(_netG, _netD, opt, device).train()
end_time = time.time()
total_time = end_time-start_time
print('Time: %.3f s/ %.3f h'%(total_time, total_time/3600))
