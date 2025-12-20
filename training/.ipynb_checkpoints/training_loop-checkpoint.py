"""Main training loop."""
import os
import csv
import time
import copy
import json
import pickle
import numpy as np
import torch
import dnnlib
import random
from torch import autocast
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from models.ldm.util import instantiate_from_config
from torch_utils.download_util import check_file_by_key

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion) 
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')

    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:         # models from EDM
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        net.sigma_min = 0.002
        net.sigma_max = 80.0
    elif dataset_name in ['lsun_bedroom']:                                  # models from Consistency Models
        from models.cm.cm_model_loader import load_cm_model
        from models.networks_edm import CMPrecond
        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
    else:
        if guidance_type == 'cg':            # clssifier guidance           # models from ADM
            assert classifier_path is not None
            from models.guided_diffusion.cg_model_loader import load_cg_model
            from models.networks_edm import CGPrecond
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
        elif guidance_type in ['uncond', 'cfg']:                            # models from LDM
            from omegaconf import OmegaConf
            from models.networks_edm import CFGPrecond
            if dataset_name in ['lsun_bedroom_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ms_coco']:
                assert guidance_type == 'cfg'
                config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=4, guidance_rate=guidance_rate, guidance_type='classifier-free', label_dim=True).to(device)
    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    net.eval()

    return net

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    pred_kwargs         = {},       # Options for predictor.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = None,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 20,       # Training duration, measured in thousands of training images.
    kimg_per_tick       = 1,        # Interval of progress prints.
    snapshot_ticks      = 1,        # How often to save network snapshots, None = disable.
    state_dump_ticks    = 20,       # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    dataset_name        = None,
    prompt_path         = None,
    guidance_type       = None,
    guidance_rate       = 0.,
    device              = torch.device('cuda'),
    **kwargs,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    
    if dataset_name in ['ms_coco']:
        # Loading MS-COCO captions for FID-30k evaluaion
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path, _ = check_file_by_key('prompts')
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)
    
    # Load pre-trained diffusion model.
    if dist.get_rank() != 0:
        torch.distributed.barrier()     # rank 0 goes first
    
    # Load pre-trained diffusion models.
    net = create_model(dataset_name, guidance_type, guidance_rate, device)
    
    if dist.get_rank() == 0:
        torch.distributed.barrier()     # other ranks follow
    
    # Construct predictor.
    dist.print0('Constructing predictor...')
    pred_kwargs.update(img_resolution=net.img_resolution)
    predictor = dnnlib.util.construct_class_by_name(**pred_kwargs) # subclass of torch.nn.Module
    predictor.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_kwargs.update(num_steps=pred_kwargs.num_steps, sampler_stu=pred_kwargs.sampler_stu, sampler_tea=pred_kwargs.sampler_tea, \
                       M=pred_kwargs.M, schedule_type=pred_kwargs.schedule_type, schedule_rho=pred_kwargs.schedule_rho, \
                       afs=pred_kwargs.afs, max_order=pred_kwargs.max_order, sigma_min=net.sigma_min, sigma_max=net.sigma_max, \
                       predict_x0=pred_kwargs.predict_x0, lower_order_final=pred_kwargs.lower_order_final)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=predictor.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    if kwargs['cos_lr_schedule']:
        dist.print0('Using Cosince Annealing Learning Rate')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_kimg * 1000//batch_size) + 1, eta_min=0.01)
    ddp = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
    
    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Generate latents and conditions in every first step
        latents = loss_fn.sigma_max * torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        labels = c = uc = None
        if net.label_dim:
            if guidance_type == 'cg':                                           # ADM models
                labels = torch.randint(net.label_dim, size=(batch_gpu,), device=device)
            elif guidance_type == 'cfg' and dataset_name in ['ms_coco']:        # Stable Diffusion (SD) models
                prompts = random.sample(sample_captions, batch_gpu)
                uc = None
                if guidance_rate != 1.0:
                    uc = net.model.get_learned_conditioning(batch_gpu * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = net.model.get_learned_conditioning(prompts)
            else:                                                               # EDM models
                labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_gpu], device=device)]

        # Generate teacher trajectories in every first step
        with torch.no_grad():
            if guidance_type in ['uncond', 'cfg']:      # LDM and SD models
                with autocast("cuda"):
                    with net.model.ema_scope():
                        teacher_traj = loss_fn.get_teacher_traj(net=net, tensor_in=latents, labels=labels, condition=c, unconditional_condition=uc)
            else:
                teacher_traj = loss_fn.get_teacher_traj(net=net, tensor_in=latents, labels=labels)

        # Perform training step by step
        for step_idx in range(loss_fn.num_steps - 1):
            optimizer.zero_grad(set_to_none=True)
            # Calculate loss
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    if guidance_type in ['uncond', 'cfg']:      # LDM and SD models
                        with net.model.ema_scope():
                            loss, str2print, stu_out = loss_fn(predictor=ddp, net=net, tensor_in=latents, 
                                                               labels=labels, step_idx=step_idx, teacher_out=teacher_traj[step_idx], condition=c, unconditional_condition=uc)
                    else:
                        loss, str2print, stu_out = loss_fn(predictor=ddp, net=net, tensor_in=latents, labels=labels, step_idx=step_idx, teacher_out=teacher_traj[step_idx])
            
                    lr = optimizer.param_groups[0]['lr']
                    lr_str = f"| lr : {lr:.8f} "
                    dist.print0(str2print + lr_str)
                    loss.sum().mul(1 / (batch_gpu_total)).backward(retain_graph=True) 

            # Update weights.
            for param in predictor.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer.step()
            latents = stu_out            

        if kwargs['cos_lr_schedule']:
            scheduler.step()
        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))
        
        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')
            
        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick > 0:
            data = dict(model=predictor, loss_fn=loss_fn)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)                  
            del data # conserve memory

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
