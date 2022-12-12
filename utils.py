import argparse
import torch

from attacks import PGD, Const,Personolize_Attack
import torch.backends.cudnn as cudnn
import random
from TartanVO import TartanVO
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset, \
    MultiTrajFolderDatasetCustom, MultiTrajFolderDatasetRealData
from torch.utils.data import DataLoader
from os import mkdir, makedirs
from os.path import isdir
from loss import VOCriterion
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='HRL')

    # run params
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: random)')
    parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
    parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--save_pose', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--save_imgs', action='store_true', default=False, help='save images (default: False)')
    parser.add_argument('--save_best_pert', action='store_true', default=False, help='save best pert (default: False)')
    parser.add_argument('--save_csv', action='store_true', default=False, help='save results csv (default: False)')

    # data loader params
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--test-dir', default="15s_closed_loop",
                        help='test trajectory folder where the RGB images are (default: None)')
    parser.add_argument('--processed_data_dir', default=None,
                        help='folder to save processed dataset tensors (default: None)')
    parser.add_argument('--preprocessed_data', action='store_true', default=False,
                        help='use preprocessed data in processed_data_dir (default: False)')
    parser.add_argument('--max_traj_len', type=int, default=8,
                        help='maximal amount of frames to load in each trajectory (default: 500)')
    parser.add_argument('--max_traj_num', type=int, default=10,
                        help='maximal amount of trajectories to load (default: 100)')
    parser.add_argument('--max_traj_datasets', type=int, default=5,
                        help='maximal amount of trajectories datasets to load (default: 10)')

    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--custom_data', action='store_true', help='custom data set (default: False)')

    # VO model params

    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--model_type', default='tartanvo', type=str, metavar='ATT', help='attack type')

    #pert_path = "/home/tal.peer/hw4_code/results/kitti_custom/tartanvo_1914/VO_adv_project_train_dataset_8_frames/train_attack/universal_attack/gradient_ascent/attack_personalize_attack_norm_Linf/opt_whole_trajectory/opt_t_crit_rms_factor_1_0_rot_crit_quat_product_factor_1_0_flow_crit_mse_factor_1_0_target_t_crit_partial_rms_factor_1_0/eval_rms/eps_1_attack_iter_50_alpha_0_7_test-run/open_loop_eval_fold_idx_0/adv_best_pert/adv_best_pert.png"
    # adversarial attacks params
    parser.add_argument('--attack', default='none', type=str, metavar='ATT', help='attack type')
    parser.add_argument('--attack_norm', default='Linf', type=str, metavar='ATTL', help='norm used for the attack')
    parser.add_argument('--attack_k', default=100, type=int, metavar='ATTK', help='number of iterations for the attack')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--attack_targeted', action='store_true', default=False, help='use targeted attacks')
    parser.add_argument('--attack_eval_mean_partial_rms', action='store_true', default=False, help='use mean partial rms criterion for attack evaluation criterion (default: False)')
    parser.add_argument('--attack_t_crit', default="none", type=str, metavar='ATTTC', help='translation criterion type for optimizing the attack (default: RMS between poses)')
    parser.add_argument('--attack_rot_crit', default="none", type=str, metavar='ATTRC', help='rotation criterion type for optimizing the attack (default: None)')
    parser.add_argument('--attack_flow_crit', default="none", type=str, metavar='ATTFC', help='optical flow criterion type for optimizing the attack (default: None)')
    parser.add_argument('--attack_target_t_crit', default="none", type=str, metavar='ATTTTC',
                        help='targeted translation criterion target for optimizing the attack, type is the same as untargeted criterion (default: None)')
    parser.add_argument('--attack_t_factor', type=float, default=1.0, help='factor for the translation criterion of the attack (default: 1.0)')
    parser.add_argument('--attack_rot_factor', type=float, default=1.0, help='factor for the rotation criterion of the attack (default: 1.0)')
    parser.add_argument('--attack_flow_factor', type=float, default=1.0, help='factor for the optical flow criterion of the attack (default: 1.0)')
    parser.add_argument('--attack_target_t_factor', type=float, default=1.0, help='factor for the targeted translation criterion of the attack (default: 1.0)')
    parser.add_argument('--window_size', type=int, default=None, metavar='WS',
                        help='Trajectory window size for testing and optimizing attacks (default: whole trajectory)')
    parser.add_argument('--window_stride', type=int, default=None, metavar='WST',
                        help='Trajectory window stride for optimizing attacks (default: whole window)')
    parser.add_argument('--load_attack', default=None, help='path to load previously computed perturbation (default: "")')
    parser.add_argument('--run_name', default='test-run',
                        help='name of run for graphs. cannot have any "_" in it!!! (default: "test-run")')

    args = parser.parse_args()
    # print("args")
    # print(args)

    return args


def compute_run_args(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("args.gpus")
    print(args.gpus)
    print("args.force_cpu")
    print(args.force_cpu)
    print("torch.cuda.is_available()")
    print(torch.cuda.is_available())
    if args.gpus is not None and not args.force_cpu and torch.cuda.is_available():
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.enabled = True
        cudnn.benchmark = True
        args.device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = 'cpu'
    print('Running inference on device \"{}\"'.format(args.device))
    torch.multiprocessing.set_sharing_strategy('file_system')

    return args


def compute_data_args(args):
    # load trajectory data from a folder
    args.test_dir_name = args.test_dir
    args.test_dir = './data/' + args.test_dir
    if args.processed_data_dir is None:
        args.processed_data_dir = "data/" + args.test_dir_name + "_processed"
    args.datastr = 'kitti_custom'
    args.focalx, args.focaly, args.centerx, args.centery = dataset_intrinsics('kitti')
    args.transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    args.focalx = 320.0/np.tan(np.pi/4.5)
    args.focaly = 320.0/np.tan(np.pi/4.5)
    args.centerx = 320.0
    args.centery = 240.0
    args.dataset_class = MultiTrajFolderDatasetCustom
    args.testDataset = \
        args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                     preprocessed_data=args.preprocessed_data,
                                     transform=args.transform, data_size=(args.image_height, args.image_width),
                                     focalx=args.focalx, focaly=args.focaly,
                                     centerx=args.centerx, centery=args.centery, max_traj_len=args.max_traj_len,
                                     max_dataset_traj_num=args.max_traj_num,
                                     max_traj_datasets=args.max_traj_datasets)

    args.testDataloader = DataLoader(args.testDataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.worker_num)

    args.traj_len = args.testDataset.traj_len
    args.traj_datasets = args.testDataset.datasets_num

    return args


def compute_VO_args(args):
    args.model = TartanVO(args.model_name, args.device)
    print("initializing attack optimization criterion")
    args.att_criterion = VOCriterion(t_crit=args.attack_t_crit,
                                     rot_crit=args.attack_rot_crit,
                                     flow_crit=args.attack_flow_crit,
                                     target_t_crit=args.attack_target_t_crit,
                                     t_factor=args.attack_t_factor,
                                     rot_factor=args.attack_rot_factor,
                                     flow_factor=args.attack_flow_factor,
                                     target_t_factor=args.attack_target_t_factor)
    args.att_criterion_str = args.att_criterion.criterion_str
    print("initializing RMS test criterion")
    args.rms_crit = VOCriterion(t_crit='rms_crit')
    print("initializing mean partial RMS test criterion")
    args.mean_partial_rms_crit = VOCriterion(t_crit="mean_partial_rms")
    print("initializing targeted RMS test criterion")
    args.target_rms_crit = VOCriterion(target_t_crit="patch", t_factor=0)
    print("initializing targeted mean partial RMS test criterion")
    args.target_mean_partial_rms_crit = VOCriterion(t_crit="mean_partial_rms", target_t_crit="patch", t_factor=0)
    args.criterions = [args.rms_crit, args.mean_partial_rms_crit,
                       args.target_rms_crit, args.target_mean_partial_rms_crit]
    args.criterions_names = ["rms_crit", "mean_partial_rms_crit",
                             "target_rms_crit", "target_mean_partial_rms_crit"]

    if args.window_size is not None:
        if args.traj_len <= args.window_size:
            args.window_size = None
        elif args.window_stride is None:
            args.window_stride = args.window_size

    return args


def compute_attack_args(args):
    if args.attack_eval_mean_partial_rms:
        print("attack evaluation criterion is mean partial RMS test criterion")
        args.att_eval_criterion = args.mean_partial_rms_crit
        args.attack_eval_str = "mean_partial_rms"
    else:
        print("attack evaluation criterion is RMS test criterion")
        args.att_eval_criterion = args.rms_crit
        args.attack_eval_str = "rms"
    if args.attack == 'const' and args.load_attack is None:
        args.attack = None

    load_pert_transform = None
    if args.load_attack is not None:
        print("loading pre-computed attack from path: " + args.load_attack)
        load_pert_transform = Compose([CropCenter((args.image_height, args.image_width)), ToTensor()])

    attack_dict = {'pgd': PGD, 'const': Const,'personalize_attack':Personolize_Attack}
    args.attack_name = args.attack
    if args.attack not in attack_dict:
        args.attack = None
        args.attack_obj = Const(args.model, args.att_eval_criterion,
                                      norm=args.attack_norm,
                                      data_shape=(args.traj_len - 1, args.image_height, args.image_width),
                                      default_pert_I1=True)

    else:
        args.attack = attack_dict[args.attack]
        if args.attack_name == 'const':
            const_pert_transform = Compose([CropCenter((args.image_height, args.image_width)), ToTensor()])
            args.attack_obj = args.attack(args.model, args.att_eval_criterion,
                                          norm=args.attack_norm,
                                          data_shape=(args.traj_len - 1, args.image_height, args.image_width),
                                          pert_path=args.load_attack,
                                          pert_transform=const_pert_transform)
        else:
            args.attack_obj = args.attack(args.model, args.att_criterion, args.att_eval_criterion,
                                          norm=args.attack_norm,
                                          data_shape=(args.traj_len - 1, args.image_height, args.image_width),
                                          n_iter=args.attack_k, alpha=args.alpha, rand_init=True,
                                          sample_window_size=args.window_size,
                                          sample_window_stride=args.window_stride,
                                          init_pert_path=args.load_attack,
                                          init_pert_transform=load_pert_transform)

    return args


def compute_output_dir(args):
    dataset_name = args.datastr
    model_name = args.model_name.split('.')[0]

    args.output_dir = 'results/' + dataset_name + '/' + model_name

    args.output_dir += '/' + args.test_dir_name
    if args.attack is None:
        args.output_dir += "/clean"
        if not isdir(args.output_dir):
            makedirs(args.output_dir)

    else:
        test_name = "train_attack"
        attack_type = "universal_attack"
        args.output_dir += '/' + test_name + '/' + attack_type
        if not isdir(args.output_dir):
            makedirs(args.output_dir)

        if args.attack_name == 'const':
            attack_name_str = "attack_" + args.attack_name + \
                              "_img_" + str(args.load_attack).split('/')[-1].split('.')[0]

            args.output_dir += "/" + attack_name_str
            if not isdir(args.output_dir):
                mkdir(args.output_dir)

        else:
            attack_window_string = "opt_whole_trajectory"
            if args.window_size is not None:
                attack_window_string = "opt_trajectory_window_size_" + str(args.window_size) + \
                                       "_stride_" + str(args.window_stride)
            attack_opt_string = 'gradient_ascent'
            attack_name_str = "attack_" + args.attack_name + "_norm_" + args.attack_norm

            args.output_dir += '/' + attack_opt_string
            if not isdir(args.output_dir):
                mkdir(args.output_dir)

            args.output_dir += "/" + attack_name_str
            if not isdir(args.output_dir):
                mkdir(args.output_dir)
            args.output_dir += '/' + attack_window_string
            if not isdir(args.output_dir):
                mkdir(args.output_dir)

            args.output_dir += '/opt_' + args.att_criterion_str
            if not isdir(args.output_dir):
                mkdir(args.output_dir)
            args.output_dir += '/eval_' + args.attack_eval_str
            if not isdir(args.output_dir):
                mkdir(args.output_dir)

            args.output_dir += "/eps_" + str(args.eps).replace('.', '_') + \
                               "_attack_iter_" + str(args.attack_k) + \
                               "_alpha_" + str(args.alpha).replace('.', '_')
            args.output_dir += '_' + str(args.run_name)
            if not isdir(args.output_dir):
                mkdir(args.output_dir)

    args.flowdir = None
    args.pose_dir = None
    args.img_dir = None
    args.adv_img_dir = None
    args.adv_pert_dir = None
    if args.save_flow:
        args.flowdir = args.output_dir+'/flow'
        if not isdir(args.flowdir):
            mkdir(args.flowdir)
    if args.save_pose:
        args.pose_dir = args.output_dir + '/pose'
        if not isdir(args.pose_dir):
            mkdir(args.pose_dir)
    if args.save_imgs:
        args.img_dir = args.output_dir + '/clean_images'
        if not isdir(args.img_dir):
            mkdir(args.img_dir)
        args.adv_img_dir = args.output_dir + '/adv_images'
        if not isdir(args.adv_img_dir):
            mkdir(args.adv_img_dir)
        args.adv_pert_dir = args.output_dir + '/adv_pert'
        if not isdir(args.adv_pert_dir):
            mkdir(args.adv_pert_dir)
    if args.save_best_pert:
        args.adv_best_pert_dir = args.output_dir + '/adv_best_pert'
        if not isdir(args.adv_best_pert_dir):
            mkdir(args.adv_best_pert_dir)
    args.save_results = args.save_flow or args.save_pose or args.save_imgs or args.save_best_pert or args.save_csv

    print('==> Will write outputs to {}'.format(args.output_dir))
    return args


def get_args():

    args = parse_args()
    args = compute_run_args(args)
    args = compute_data_args(args)
    args = compute_VO_args(args)
    args = compute_attack_args(args)
    args = compute_output_dir(args)

    print("arguments parsing finished")
    return args





