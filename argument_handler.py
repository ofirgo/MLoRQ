import argparse

from compression.configs.compression_config import (ThresholdMethod, ABSPLIT, ParetoCost, SVDScores)


def argument_handler():
    #################################
    ######### Run Arguments #########
    #################################

    # Settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', '-m', type=str, default="vit_s",
                        help='The name of the model to run')
    parser.add_argument('--model_type', type=str, default='vision', choices=['vision'])

    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=1024)

    parser.add_argument('--eval_float_accuracy', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    # Quantization
    parser.add_argument('--weight_n_bits', type=float, default=8,
                        help="For Mixed Rank and Quantization this is the avg weight bit.")
    parser.add_argument('--bit_options', nargs='+', type=float, default=[2, 3, 4, 6, 8])

    # Low Rank
    parser.add_argument('--ab_split_method', type=str, default='U_SV',
                        choices=[e.value for e in ABSPLIT])

    # Weight Quantization
    parser.add_argument('--per_channel_Ar', action='store_false', default=True)
    parser.add_argument('--per_channel_Br', action='store_false', default=True)
    parser.add_argument('--threshold_method', type=str, default='HMSE',
                        choices=[i.name for i in ThresholdMethod])

    # Activation Quantization
    parser.add_argument('--disable_activation_quantization', action='store_true', default=False)
    parser.add_argument('--activation_n_bits', type=int, default=8)
    parser.add_argument('--act_num_samples', type=int, default=32)
    parser.add_argument('--activation_mp', action='store_true', default=False)
    parser.add_argument('--disable_ln_reparam', action='store_true', default=False)
    parser.add_argument('--disable_softmax_log_scale', action='store_true', default=False)
    parser.add_argument('--disable_ridge_regression', action='store_true', default=False)
    parser.add_argument('--ridge_regression_num_samples', type=int, default=32)

    # Hessians
    parser.add_argument('--h_n_iters', type=int, default=100)
    parser.add_argument('--h_w_num_samples', type=int, default=32)
    parser.add_argument('--weighted_svd_scores', type=str, default='LFH',
                        choices=[i.name for i in SVDScores])

    # MRaP Search
    parser.add_argument('--pareto_cost', type=str, default='HMSEPerOutChannel',
                        choices=[i.name for i in ParetoCost])
    parser.add_argument('--num_inter_points', type=int, default=5)

    parser.add_argument('--disable_finetune', action='store_true', default=False)
    parser.add_argument('--finetune_iters', type=int, default=20000)
    parser.add_argument('--finetune_batch_size', type=int, default=32)
    parser.add_argument('--finetune_lr', type=float, default=0.3)
    parser.add_argument('--reg_factor', type=float, default=0.3)

    args = parser.parse_args()

    args.pareto_cost = ParetoCost(args.pareto_cost)
    args.ab_split_method = ABSPLIT(args.ab_split_method)
    args.threshold_method = ThresholdMethod(args.threshold_method)
    args.weighted_svd_scores = SVDScores(args.weighted_svd_scores)

    return args
