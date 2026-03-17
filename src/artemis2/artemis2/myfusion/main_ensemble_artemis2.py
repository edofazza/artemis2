import os

os.environ['MPLCONFIGDIR'] = '/workdir/.matplotlib_local'
os.environ['HF_HOME'] = '/workdir/.cache'

import string
import torch
import numpy as np
import random
import time
from ensemble.ensemble import Ensemble
from ensemble.genetic import GeneticEnsemble
#from models.msqnet_variation1 import MSQNetVariation1
#from models.msqnet_variation1contrastive import MSQNetVariation1Contrastive
from models.videomae_artemis_contrastive import VideoMAEArtemisContrastive


#from fvcore.nn import FlopCountAnalysis
#from fvcore.nn import flop_count_table

def main(args):
    if (args.seed >= 0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("[INFO] Setting SEED: " + str(args.seed))
    else:
        print("[INFO] Setting SEED: None")

    if (torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.", flush=True)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type:", str(device), flush=True)

    config = dict()
    config['path_dataset'] = '.'  # MODIFIED, substituted get_config
    if args.dataset == "animalkingdom":
        dataset = 'AnimalKingdom'
    elif args.dataset == "baboonland":
        dataset = 'baboonland'
    elif args.dataset == "mammalnet":
        dataset = 'mammalnet'
    else:
        dataset = string.capwords(args.dataset)
    path_data = os.path.join(config['path_dataset'], dataset)
    print("[INFO] Dataset path:", path_data, flush=True)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    if args.train:
        # NO TRANSFORMATION
        train_transform = manager.get_test_transforms()
        data_loader = manager.get_cross_loader(train_transform, args.k)
        print(f"[INFO] Cross val {args.k} size:", str(len(data_loader.dataset)), flush=True)
    else:
        # test data
        val_transform = manager.get_test_transforms()
        data_loader = manager.get_test_loader(val_transform)
        print("[INFO] Test size:", str(len(data_loader.dataset)), flush=True)

    # MODEL DEFINITIONS
    model1 = VideoMAEArtemisContrastive(class_embed=torch.rand((num_classes, 768)),  # ALIGN text cosine
                                        num_frames=16,
                                        recurrent='conv',
                                        fusion='normal',
                                        residual=True,
                                        relu=False,
                                        summary_residual=False,
                                        backbone_residual=True,
                                        linear2_residual=False,
                                        image_residual=True).to(device)
    #model1.load_state_dict(torch.load(
    #    'artemis2_model_mammalnet_cosine/artemis_vm_dino_s4_align_cosine_variation1_conv_residual_backboneresidual_imageresidual_.pth'))
    #model1.load_state_dict(torch.load('models/variation1_conv_residual_backboneresidual_imageresidual_.pth'))
    model1.load_state_dict(torch.load('artemis2_model_baboonland_cosine/artemis2_align_cosine_variation1_conv_residual_backboneresidual_imageresidual_.pth'))

    model2 = VideoMAEArtemisContrastive(class_embed=torch.rand((num_classes, 768)),  # FLAVA text cosine
                                        num_frames=16,
                                        recurrent='conv',
                                        fusion='normal',
                                        residual=True,
                                        relu=False,
                                        summary_residual=False,
                                        backbone_residual=True,
                                        linear2_residual=False,
                                        image_residual=True).to(device)
    """model2.load_state_dict(
        torch.load(
            'artemis2_model_mammalnet_cosine/artemis_vm_dino_s4_flava_cosine_variation1_conv_residual_backboneresidual_imageresidual_.pth'))"""
    model1.load_state_dict(torch.load(
        'artemis2_model_baboonland_cosine/artemis2_flava_cosine_variation1_conv_residual_backboneresidual_imageresidual_.pth'))

    model3 = VideoMAEArtemisContrastive(class_embed=torch.rand((num_classes, 768)),  # ALIGN var2 cosine
                                        num_frames=16,
                                        recurrent='bilstm',
                                        fusion='normal',
                                        residual=True,
                                        relu=False,
                                        summary_residual=True,
                                        backbone_residual=True,
                                        linear2_residual=True,
                                        image_residual=True).to(device)
    #model3.load_state_dict(
    #    torch.load(
    #        'artemis2_model_mammalnet_cosine/artemis_bilstm_vm_dino_s4_align_best_cosine_variation1_bilstm_residual_sumresidual_backboneresidual_linear2residual_imageresidual_.pth'))
    #model3.load_state_dict(
    #    torch.load('models/variation1_bilstm_residual_sumresidual_backboneresidual_linear2residual_imageresidual_.pth'))
    model3.load_state_dict(torch.load('artemis2_model_baboonland_cosine/artemis2_align_bilstm_cosine_variation1_bilstm_residual_sumresidual_backboneresidual_linear2residual_imageresidual_.pth'))

    models = [model1, model2, model3]
    """print('********* MODEL 1 *********')
    s = time.time()
    flops = FlopCountAnalysis(model1, (torch.rand(1, 16, 3, 224, 224).to('cuda', non_blocking=True),
                                       torch.rand(1, 512).to('cuda', non_blocking=True)))
    print(time.time() - s)
    print(flop_count_table(flops, max_depth=1))

    print('\n\n********* MODEL 2 *********')
    s = time.time()
    flops = FlopCountAnalysis(model2, (
    torch.rand(1, 16, 3, 224, 224).to('cuda', non_blocking=True), torch.rand(1, 512).to('cuda', non_blocking=True)))
    print(time.time() - s)
    print(flop_count_table(flops, max_depth=1))

    print('\n\n********* MODEL 3 *********')
    s = time.time()
    flops = FlopCountAnalysis(model3, (
        torch.rand(1, 16, 3, 224, 224).to('cuda', non_blocking=True),
        torch.rand(1, 512).to('cuda', non_blocking=True)))
    print(time.time() - s)
    print(flop_count_table(flops, max_depth=1))"""
    ens = Ensemble(models=models, data_loader=data_loader, device=device, num_labels=num_classes)

    if args.type == 'ensemble':
        initial_time = time.time()
        #eval = ens.test(weights=[0.35527880632037867, 0.33102491647831506, 0.31369627720130633])    # FOLD 0 AK
        # eval = ens.test(weights=[0.043700000830110684, 0.557841995937574, 0.39845800323231534])  # FOLD 0 ARTEMIS 2 AK
        #eval = ens.test(weights=[0.4247185529422919, 0.5141022914447522, 0.06117915561295602])  # FOLD 0 ARTEMIS 2 Mammalnet
        """eval = ens.test(weights=[0.3461496640573008, 0.1274038002556779, 0.5264465356870214])  # FOLD 0 ARTEMIS 2 Baboonland
        final_time = time.time()
        print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)"""

        #eval = ens.test(weights=[0.34381520529054543, 0.35082624709176513, 0.3053585476176895])  # FOLD 1 AK
        #eval = ens.test(weights=[0.05653022001811124, 0.5656833830569415, 0.37778639692494725])  # FOLD 1 ARTEMIS 2 AK
        #eval = ens.test(weights=[0.2908058851918934, 0.5044399923036108, 0.20475412250449568])  # FOLD 1 ARTEMIS 2 Mammalnet
        """initial_time = time.time()
        eval = ens.test(weights=[0.42376538893844656, 0.021070675591464996, 0.5551639354700885])  # FOLD 1 ARTEMIS 2 Baboonland
        final_time = time.time()
        print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)"""

        #eval = ens.test(weights=[0.34348380125277234, 0.3341105545108527, 0.32240564423637497])  # FOLD 2 AK
        #eval = ens.test(weights=[0.12102572448618919, 0.4717734894043526, 0.4072007861094582])  # FOLD 2 ARTEMIS 2 AK
        #eval = ens.test(weights=[0.21255837564721652, 0.4128642317334276, 0.37457739261935585])  # FOLD 2 ARTEMIS 2 Mammalnet
        initial_time = time.time()
        eval = ens.test(weights=[0.48978215347165427, 0.06864545765898786, 0.4415723888693579])  # FOLD 2 ARTEMIS 2 Baboonland
        final_time = time.time()
        print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)

        #eval = ens.test(weights=[0.3578997689238905, 0.3847545988164762, 0.25734563225963336])  # FOLD 3 AK
        #eval = ens.test(weights=[0.09194458938883218, 0.4827473656891539, 0.425308044922014])  # FOLD 3 ARTEMIS 2 AK
        """initial_time = time.time()
        #eval = ens.test(weights=[0.38374319002861157, 0.11352490890743701, 0.5027319010639514])  # FOLD 3 ARTEMIS 2 Mammalnet
        eval = ens.test(weights=[0.33916547862735424, 0.23979996807575232, 0.4210345532968935])  # FOLD 3 ARTEMIS 2 Baboonland
        final_time = time.time()
        print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)"""

        #eval = ens.test(weights=[0.3468346961521773, 0.3666702635596512, 0.2864950402881715])  # FOLD 4 AK
        #eval = ens.test(weights=[0.09963801121109825, 0.4822334857142855, 0.41812850307461635])  # FOLD 4 ARTEMIS 2 AK
        """initial_time = time.time()
        #eval = ens.test(weights=[0.21255837564721652, 0.4128642317334276, 0.37457739261935585])  # FOLD 4 ARTEMIS 2 Mammalnet
        eval = ens.test(weights=[0.5703018330678843, 0.02741343456953997, 0.40228473236257584])  # FOLD 4 ARTEMIS 2 Baboonland
        final_time = time.time()
        print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)

        eval = ens.test(weights=[1.0, 1.0, 1.0])
        final_time = time.time()
        print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)"""
    elif args.type == 'ga':
        initial_time = time.time()
        genetic_ensemble = GeneticEnsemble(ens, 10, 0.5, 0.2, 2, fold=args.k, pop_size=15)
        genetic_ensemble.train()
        final_time = time.time()
        print("[INFO] Evaluation Time: {:.2f}".format(final_time - initial_time), flush=True)
    elif args.type == 'rl_static':
        from ensemble.reinforcement_static import train
        train(ens)
    elif args.type == 'rl_dynamic':
        from ensemble.reinforcement_dynamic import train, test
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(args.seed)
        if args.train:
            train(models, data_loader, device, args.k)
        else:
            test(models, data_loader, device)
    else:
        print("[ERROR] Unknown type")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training script for ensemble")
    parser.add_argument('--type', type=str, default='ensemble', help="ensemble/ga/rl")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--dataset", default='animalkingdom', type=str, help='animalkingdom')
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default=16, type=int, help="Size of the mini-batch")
    parser.add_argument("--num_workers", default=2, type=int,
                        help="Number of torchvision workers used to load data (default: 2)")
    parser.add_argument("--distributed", default=False, type=bool, help="Distributed training flag")
    parser.add_argument("--train", action='store_true', help="train/test")
    parser.add_argument("--k", default=0, type=int, help="set between 0 and 4 (included)")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    args = parser.parse_args()

    main(args)
