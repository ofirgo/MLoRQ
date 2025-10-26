import torch
from tqdm import tqdm

from argument_handler import argument_handler
from compression.configs.compression_config import CompressionConfig, ThresholdMethod, SVDScores, ParetoCost
from compression.module.prepare_module import prepare_module
from compression.quantization.activation_quantization import activation_quantization_param_search
from compression.quantization.activation_quantization import insert_activation_quantization, \
    ln_reparameterization, hook_fp_act, replace_W
from compression.quantization.finetune import FineTuning
from compression.solver import run_solver
from compression.weighted_svd.weights_lfh import set_model_hessian_scores
from constants import LINEAR_QUANTIZE_OPERATORS
from helpers.set_seed import set_seed
from helpers.utils import is_compressed_layer, is_quantized_activation
from model_managers.base_model import BaseModel, ModelManagerArgs
from model_managers.vision_model import VisionModel


def get_model_manager(**kwargs) -> BaseModel:
    model_type_map = {
        "vision": VisionModel
    }

    model_manager_args = ModelManagerArgs(
        model_name=kwargs["model_name"],
        batch_size=kwargs["batch_size"],
        num_samples=kwargs["num_samples"],
        train_dir=kwargs["train_data_path"],
        val_dir=kwargs["val_data_path"]
    )
    model_type = kwargs['model_type']
    model_manager = model_type_map[model_type]
    return model_manager(model_manager_args)


def register_input_shapes_hook(module, name, handles):
    def hook(module, inputs, outputs):
        if is_compressed_layer(module) and not hasattr(module, 'input_shape'):
            module.input_shape = inputs[0].shape[1:]  # Ignore batch size

    return module.register_forward_hook(hook)


def compute_float_references(compressed_model, representative_dataset):
    handles = []
    for name, layer in compressed_model.named_modules():
        if is_compressed_layer(layer):
            handle = register_input_shapes_hook(layer, name, handles)
            handles.append(handle)

    output_ref = {}  # batch_idx --> layer --> output tensor
    for batch_idx, batch in enumerate(representative_dataset):
        data = model_manager.data_to_device(batch)
        batch_output_ref = model_manager.forward(compressed_model, data)
        output_ref[batch_idx] = {k: v.cpu() for k, v in batch_output_ref.items()} \
            if isinstance(batch_output_ref, dict) else batch_output_ref.cpu()

    for h in handles:
        h.remove()

    return output_ref


if __name__ == '__main__':

    #####################
    ######## Init #######
    #####################

    args = argument_handler()
    set_seed(args.seed)

    model_manager = get_model_manager(**vars(args))
    float_model = model_manager.float_model
    float_model.eval()
    val_data_loader = model_manager.get_validation_data_loader()
    float_accuracy = model_manager.float_accuracy
    if args.eval_float_accuracy:
        float_accuracy = model_manager.evaluate(float_model, val_data_loader)
        model_manager.set_float_accuracy(float_accuracy)

    cc = CompressionConfig(weight_bit_list=args.bit_options,
                           ab_split=args.ab_split_method,
                           per_channel_Ar=args.per_channel_Ar,
                           per_channel_Br=args.per_channel_Br,
                           pareto_cost=args.pareto_cost,
                           svd_scores=args.weighted_svd_scores,
                           threshold_method=args.threshold_method,
                           num_inter_points=args.num_inter_points,
                           activation_n_bits=args.activation_n_bits,
                           activation_mp=args.activation_mp,
                           disable_softmax_log_scale=args.disable_softmax_log_scale,
                           disable_ln_reparam=args.disable_ln_reparam)

    #######################
    ##### Load Dataset ####
    #######################
    representative_dataset = model_manager.get_representative_dataset(args.num_samples, False, False)

    weight_n_bits = args.weight_n_bits

    compressed_model, float_model = prepare_module(float_model, model_manager, cc)

    # Compute float output reference
    with torch.no_grad():
        output_ref = compute_float_references(compressed_model, representative_dataset)

    ######################
    #### Init Hessian ####
    ######################
    compute_hessians = (args.threshold_method == ThresholdMethod.HMSE
                        or args.pareto_cost == ParetoCost.HMSEPerOutChannel
                        or cc.svd_scores == SVDScores.LFH)
    if compute_hessians:
        h_num_samples = args.h_w_num_samples
        batch = next(iter(representative_dataset))
        if isinstance(batch, dict) or type(batch).__name__ == 'BatchEncoding':
            h_images = {k: v[:h_num_samples] for k, v in batch.items()}
        else:
            h_images = batch[0][:h_num_samples]
        h_n_iter = args.h_n_iters
        set_model_hessian_scores(compressed_model, h_images, n_iter=h_n_iter)

    ##################################
    #### Init weights compression ####
    ##################################
    for n, m in tqdm(compressed_model.named_modules()):
        if is_compressed_layer(m):
            m.init_layer_compression(in_compression_config=cc,
                                     output_ref=output_ref,
                                     representative_data_loader=representative_dataset,
                                     qm=compressed_model,
                                     model_manager=model_manager)

    ###############################
    #### Prepare validation DS ####
    ###############################
    val_data_loader = model_manager.get_validation_data_loader()

    ########################
    #### Prepare Solver ####
    ########################
    compressed_model = compressed_model.to(model_manager.device)
    optimization_function = run_solver(compressed_model, cc, representative_dataset,
                                       model_manager, output_ref)

    finetune = None
    if not args.disable_finetune:
        model_manager.batch_size = args.finetune_batch_size
        finetune_repdatset = model_manager.get_representative_dataset(args.num_samples, True, True)
        model_manager.batch_size = args.batch_size
        finetune = FineTuning(finetune_repdatset, model_manager, iters=args.finetune_iters,
                              batch_size=args.finetune_batch_size,
                              lr=args.finetune_lr, reg_factor=args.reg_factor)

    # check model weights dtype for size rate calculation
    float_model_n_bits = [m.weight.dtype.itemsize for m in compressed_model.modules() if hasattr(m, 'weight')]
    if len(set(float_model_n_bits)) > 1:
        raise Exception("mixed float precision")
    float_model_n_bits = 8 * float_model_n_bits[0]

    ####################
    #### Run Solver ####
    ####################
    sol = {}  # (compressed layers index, name) --> config (bit / (bit_a, bit_b, rank)
    compression_results = optimization_function(weight_n_bits / float_model_n_bits)
    comp_layers = [(n, m) for n, m in compressed_model.named_modules() if is_compressed_layer(m)]
    for idx, (n, m) in enumerate(comp_layers):
        if compression_results[n].is_low_rank:
            sol[(idx, n)] = (compression_results[n].bit_width_quantization_a,
                             compression_results[n].bit_width_quantization_b, compression_results[n].rank)
        else:
            sol[(idx, n)] = (compression_results[n].bit_width_quantization,)

    # Recreating the compressed model
    compressed_model, _ = prepare_module(float_model, model_manager, cc)

    #################################
    #### Activation Quantization ####
    #################################
    if not args.disable_activation_quantization:
        compressed_model = insert_activation_quantization(model=compressed_model,
                                                          input_activations_quant=LINEAR_QUANTIZE_OPERATORS,
                                                          compression_config=cc)

        ##############################
        # Activation threshold search
        ##############################
        calib_samples = []
        rep_dataset = iter(representative_dataset)
        act_num_samples = min(args.act_num_samples, args.num_samples)
        num_batches = 1 if act_num_samples == args.batch_size else act_num_samples // args.batch_size + 1
        for _ in range(num_batches):
            batch_samples = model_manager.data_to_device(next(rep_dataset))
            calib_samples.append(batch_samples)
        calib_samples = torch.cat(calib_samples, dim=0)

        calib_samples_rr = []
        rep_dataset = iter(representative_dataset)
        ridge_regression_num_samples = min(args.ridge_regression_num_samples, args.num_samples)
        num_batches = 1 if ridge_regression_num_samples == args.batch_size else (
                ridge_regression_num_samples // args.batch_size + 1)
        for _ in range(num_batches):
            batch_samples = model_manager.data_to_device(next(rep_dataset))
            calib_samples_rr.append(batch_samples)
        calib_samples_rr = torch.cat(calib_samples_rr, dim=0)

        for n, m in compressed_model.named_modules():
            if is_quantized_activation(m):
                m.set_activation_quantization(True)

        compressed_model = activation_quantization_param_search(quant_model=compressed_model,
                                                                calib_samples=calib_samples[:act_num_samples],
                                                                model_manager=model_manager,
                                                                compression_config=cc)
        for n, m in compressed_model.named_modules():
            if is_quantized_activation(m):
                m.set_activation_quantization(False)

        ##############################
        # LayerNorm Reparametrization
        ##############################
        if not args.disable_ln_reparam:
            ln_reparameterization(compressed_model, cc=cc)

        ###################
        # Ridge Regression
        ##################
        if not args.disable_ridge_regression:
            for n, m in compressed_model.named_modules():
                if is_quantized_activation(m):
                    m.set_activation_quantization(False)
            fp_folder_path = hook_fp_act(compressed_model, calib_samples_rr[:args.ridge_regression_num_samples],
                                         args)

            for n, m in compressed_model.named_modules():
                if is_quantized_activation(m):
                    m.set_activation_quantization(True)

            replace_W(compressed_model, fp_folder_path)

            for n, m in compressed_model.named_modules():
                if is_quantized_activation(m):
                    m.set_activation_quantization(False)

    ###########################
    #### Recompute Hessian ####
    ###########################
    if compute_hessians:
        h_num_samples = args.h_w_num_samples
        samples, _ = next(iter(representative_dataset))
        h_images = samples[:h_num_samples]
        h_n_iter = args.h_n_iters
        set_model_hessian_scores(compressed_model, h_images, n_iter=h_n_iter)

    ########################################
    ### Recalibrate compressed model params
    ########################################
    comp_layers = [(n, m) for n, m in compressed_model.named_modules() if is_compressed_layer(m)]
    for idx, (n, m) in tqdm(enumerate(comp_layers)):
        sol_config = sol[(idx, n)]
        m.init_layer_compression(in_compression_config=cc,
                                 output_ref=output_ref,
                                 representative_data_loader=representative_dataset,
                                 qm=compressed_model,
                                 model_manager=model_manager,
                                 config_to_set=sol_config)

    ##############################
    #### Set Compressed Model ####
    ##############################
    for n, m in compressed_model.named_modules():
        if is_compressed_layer(m):
            assert len(m.compression_options.compression_options_list) == 1
            m.set_compression_config(m.compression_options.compression_options_list[0])
            m.enable_compression()

    if not args.disable_finetune:
        assert finetune is not None, "Finetune function not initialized."
        finetune(compressed_model, float_model)

    #########################
    #### Run Evaluation ####
    ########################
    if not args.disable_activation_quantization:
        for n, m in compressed_model.named_modules():
            if is_quantized_activation(m):
                m.set_activation_quantization(True)

    acc = model_manager.evaluate(compressed_model, val_data_loader)

    print("float accuracy:", float_accuracy)
    print(f"compressed accuracy: avg. bits = {weight_n_bits}, acc = {acc}")
