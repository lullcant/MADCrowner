import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from accelerate import Accelerator
import os
import random
import numpy as np
import torch.nn.functional as F
import re
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from pytorch3d.loss import chamfer_distance
from models.loss import curvature_penalty_loss
from pytorch3d.ops import sample_farthest_points
from models.SAP.src.utils import *
import pyvista as pv
from models.crowndeformer import CrownDeformer
from mydataset.DentalDataset import *
from accelerate import DataLoaderConfiguration,DistributedDataParallelKwargs
dataloader_config = DataLoaderConfiguration(split_batches=True)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(dataloader_config=dataloader_config,kwargs_handlers=[kwargs])
neg121 = False
def setup_logging(log_file):
    """
    设置日志文件和日志格式
    """
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Training started.")

def cycle(dl):
    while True:
        for data in dl:
            yield data

def train(model, train_loader, val_loader,args,log_file):
    model.to(accelerator.device) 


    # 使用 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=1e-6)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    train_loader = cycle(train_loader)
    
    
    
    
    step = 0
    if args.continue_ckpt_dir:
        match = re.search(r'_step_(\d+)\.pth', args.continue_ckpt_dir)
        if match:
            step = int(match.group(1))
            logging.info(f"Resuming training from step {step}")
      
            for _ in range(step):
                scheduler.step()
        else:
            logging.warning("Could not parse step from checkpoint filename. Starting from step 0.")
    
    best_val_dice = 0.0

    with tqdm(total=args.num_steps, initial=step, desc="Training", unit="step") as pbar:
        while step < args.num_steps:
            model.train()
           
            total_loss = 0.0
            for _ in range(args.accumulation_steps):
                
                inputs,crown_sampled,target_normal, template_tensor, psr_grid,curvatures,margin,_ = next(train_loader)
                with accelerator.autocast():
                    
                   
     
                    pred_psr,seed,coarse_pc,dense_cloud = model(inputs, template_tensor)
                 
                    if (step+1) % 20000 == 0:
                        dense_cloud_np = dense_cloud[0].detach().cpu().numpy()
                        point_cloud = pv.PolyData(dense_cloud_np[...,:3])
                 
                        output_filename = os.path.join(args.save_path[:-4], f"output_batch_training_step{step}.ply")
                        point_cloud.save(output_filename)
                     
                    psr_loss =  F.mse_loss(pred_psr,psr_grid)
        
                    crown_sample_2048 = sample_farthest_points(crown_sampled[...,:3],K=args.sample_points)[0]
 
                    loss_cpl =  curvature_penalty_loss(dense_cloud,crown_sampled,curvatures,margin,weight_lambda=args.lambda_w)
                    loss_seed =  chamfer_distance(seed,crown_sample_2048)[0]
                    loss_coarse = chamfer_distance(coarse_pc,crown_sampled)[0]
                    loss_dense = loss_cpl + loss_coarse + loss_seed
                  
                    loss = loss_dense + psr_loss 
                    loss = loss / args.accumulation_steps
                    total_loss += loss.item()
                accelerator.backward(loss)
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        print(f"{name} requires grad: {param.requires_grad}")
            
            pbar.set_description(f'loss: {total_loss:.4f} psr: {psr_loss:.4f} loss_coarse: {loss_coarse:.4f} loss_seed: {loss_dense:.4f} loss_cpl: {loss_cpl:.4f}')
            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), 1) # 1 is max grad norm
            optimizer.step()
            optimizer.zero_grad()  
            scheduler.step()
            accelerator.wait_for_everyone()    

       
            step += 1
            
       
            if step >= args.num_steps:
                break

         
            if accelerator.is_main_process:
                if step % args.validation_interval == 0 or step == args.num_steps:
                    val_dice,val_psr= validate(model, val_loader, step=step,save_path=args.save_path)
                    logging.info(f"Step [{step}/{args.num_steps}], Validation hausdorff: {val_dice:.4f} , Validation Psr :{val_psr:.4f}")
                    
                    # 保存当前步骤的模型
                    if step % args.validation_interval == 0 and step>100000:
                        valmodel = accelerator.unwrap_model(model)  # unwrap
                        step_model_path = os.path.join('checkpoints',f"{args.save_path.replace('.pth', f'_step_{step}.pth')}")
                        torch.save(valmodel.state_dict(), step_model_path)
                        logging.info(f"Saved model at step {step}")
                    
                    # 保存最优模型
                    if val_dice < best_val_dice:
                        best_val_dice = val_dice
                        valmodel = accelerator.unwrap_model(model)  # unwrap
                        best_model_path = os.path.join('checkpoints',f"{args.save_path.replace('.pth', '_best.pth')}")
                        torch.save(valmodel.state_dict(), best_model_path)
                        print(f"Saved best model at step {step}")
            pbar.update(1)
            if step % args.log_interval == 0:
          
                logging.info(f"Step {step} - Chamfer: {loss - psr_loss:.4f} - PSR: {psr_loss:.4f}")



def validate(model, val_loader, step,save_path='./chamfer_validation_outputs'):
    model.eval()
    val_hausdorff = 0.0
    val_psr = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets,target_normal, template_tensor,psr_grid,curvatures,margin, _) in enumerate(val_loader):
            
            inputs,targets = inputs.to(accelerator.device), targets.to(accelerator.device)
            
            pred_psr,seed,coarse_pc,dense_cloud = model(inputs, template_tensor)
           
            if neg121:
                inputs = (inputs + 1) /2
                dense_cloud = (dense_cloud + 1) / 2
                targets = (targets + 1) / 2
            hausdorff = chamfer_distance(10.0*dense_cloud[...,:3],10.0*targets[...,:3],point_reduction="mean")[0]
            psr_loss = F.mse_loss(pred_psr,psr_grid)
         
            val_hausdorff += hausdorff
            val_psr += psr_loss
          
          
            if batch_idx < 2:
         
                input_np = inputs.cpu().numpy()
                output_np = dense_cloud.cpu().numpy()  
                targets_np = targets.cpu().numpy()
         
                for i in range(output_np.shape[0]):
                    
                    v, f, _ = mc_from_psr(pred_psr[i], zero_level=0)
                    de_p = v * 10.0
                    mesh_out_file = os.path.join(save_path[:-4], f"mesh_batch{batch_idx+1}_sample{i+1}.ply")
                    export_mesh(mesh_out_file,de_p, f)
                    point_cloud_input = pv.PolyData(input_np[i][...,:3])
                    point_cloud_gt = pv.PolyData(targets_np[i][...,:3])
                    point_cloud = pv.PolyData(output_np[i][...,:3])  
                    output_filename = os.path.join(save_path[:-4], f"output_batch{batch_idx+1}_sample{i+1}_step{step}.ply")
                    gt_filename = os.path.join(save_path[:-4], f"output_batch{batch_idx+1}_gt{i+1}_step{step}.ply")
                    input_filename = os.path.join(save_path[:-4], f"input_batch{batch_idx+1}_input{i+1}_step{step}.ply")
                    point_cloud.save(output_filename)
                    point_cloud_gt.save(gt_filename)
                    point_cloud_input.save(input_filename)
       
                    logging.info(f"Saved: {output_filename}")
                    logging.info(f"Saved: {gt_filename}")

           

    val_hausdorff/=len(val_loader)
    val_psr /= len(val_loader)
    return val_hausdorff,val_psr

def load_data(batch_size=4, train_path='./train_data',sample_points=1024):

    train_dataset = IOS_Dataset(train_path,sample_points=sample_points)
    val_dataset = IOS_Dataset(train_path,is_train=False,sample_points=sample_points)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8)
    print(len(train_dataset))
    print(len(val_dataset))
    return train_loader, val_loader

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=1000, help='Total number of steps for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of steps for gradient accumulation')
    parser.add_argument('--save_path', type=str, default='./unet3d_model.pth', help='Path to save the trained model')
    parser.add_argument('--train_path', type=str, default='./train_data', help='Path to training data')
    parser.add_argument('--val_path', type=str, default='./val_data', help='Path to validation data')
    parser.add_argument('--validation_interval',type=int,default=4000,help='interval steps to validate')
    parser.add_argument('--continue_ckpt_dir',type=str,required=False,help='whether to use exist ckpt')
    parser.add_argument('--log_interval', type=int, default=50, help='Interval steps to log training information')
    parser.add_argument('--lambda_w',default=1,type=float,help='Weight of curvature penalty')
    parser.add_argument('--sample_points',default=1024,type=int,help='Weight of curvature penalty')
    args = parser.parse_args()
    model = CrownDeformer()
    if args.continue_ckpt_dir:
        ckpt = torch.load(args.continue_ckpt_dir)
        ckpt = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)

    train_loader, val_loader = load_data(batch_size=args.batch_size, train_path=args.train_path,sample_points=args.sample_points)
    if not os.path.exists(args.save_path[:-4]):
        os.makedirs(args.save_path[:-4])  
    log_file = os.path.join(args.save_path[:-4],'training.log')
    setup_logging(log_file)

    train(model, train_loader, val_loader, args,log_file=log_file)
if __name__== "__main__":
    seed_everything(42)
    main()