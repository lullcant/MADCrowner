import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from mydataset.DentalDataset import IOS_Datasetply4 ,IOS_Datasetply2 # 假设已经实现
import pyvista as pv
import os
import numpy as np
import random
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
#from PyTorchEMD.emd import earth_mover_distance
from pytorch3d.ops import sample_farthest_points
from models.pattn_wld import Pattn,Pattn_ablation
from scipy.spatial import distance_matrix
from models.SAP.src.utils import *
save_file = False
accelerator = Accelerator()
neg121 = False
from torch_geometric.nn import knn
def compute_fscore(x, y, fraction=0.03):
    # 动态阈值（基于 y 的边界框）
    min_xyz = y.min(dim=0)[0]
    max_xyz = y.max(dim=0)[0]
    diag_length = torch.sqrt(torch.sum((max_xyz - min_xyz) ** 2))
    threshold = fraction * diag_length

    # GT→预测的匹配 (Recall)
    dist_x_to_y = torch.cdist(x, y).min(dim=1)[0]  # x 中点到 y 的最小距离
    recall = (dist_x_to_y <= threshold).float().mean()

    # 预测→GT的匹配 (Precision)
    dist_y_to_x = torch.cdist(y, x).min(dim=1)[0]  # y 中点到 x 的最小距离
    precision = (dist_y_to_x <= threshold).float().mean()

    # F-Score
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    return fscore

def fidelity_ratio_and_point_cloud(
    x: torch.Tensor,       # 需要找最近邻的点集 (e.g. GT 点云)
    y: torch.Tensor,       # 用来做最近邻搜索的点集 (e.g. Pred 点云)
    fraction: float = 0.03 # 距离阈值 (单位 mm, 或你的坐标系单位)
):
    """
    1. 对 x 中每个点，求其在 y 中的最近邻距离。
    2. 若距离 <= threshold 则记为 inlier，否则记为 outlier。
    3. 返回 (ratio, new_point_cloud):
       - ratio = (inlier 数量) / (outlier 数量)
       - new_point_cloud: (N, 4) 张量
         [ x_coord, y_coord, z_coord, dist_to_nearest ]
    """
    min_xyz = y.min(dim=0)[0]
    max_xyz = y.max(dim=0)[0]
    diag_length = torch.sqrt(torch.sum((max_xyz - min_xyz) ** 2))
    threshold = fraction * diag_length

    # 使用 knn 找最近邻（k=1）
    # 注意 knn(x, y, k=1) 或 knn(y, x, k=1) 的查询/引用点顺序
    # 根据 PyTorch Geometric 版本不同，若结果不对可对调
    assign_index_x2y = knn(y, x, k=1, batch_size=1)
    
    # 计算 x 中各点与最近邻 y[*] 的欧几里得距离
    nearest_dists = torch.sqrt(torch.sum((x - y[assign_index_x2y[1]])**2, dim=1))
    
    # 根据 threshold 分为 inliers 与 outliers
    inliers_mask = (nearest_dists <= threshold)  # True/False
    outliers_mask = ~inliers_mask

    num_inliers = inliers_mask.sum().item()   # <= threshold 的点数
    num_outliers = outliers_mask.sum().item() # > threshold 的点数

    # 计算比值: (inlier 数量 / outlier 数量)
    if num_outliers == 0:
        ratio = 1
    else:
        ratio = num_inliers / (num_outliers+num_inliers)
    
    # 生成新的 (N, 4) 点云: 前三列是 x 的原始坐标，最后一列是与最近邻的距离
    new_point_cloud = torch.cat([x, nearest_dists.unsqueeze(1)], dim=1)
    
    return ratio, new_point_cloud

def extract_and_rename(path):
    # 分割路径为各个部分
    parts = path.split(os.sep)
    
    # 提取倒数第三块和最后一块
    third_last_part = parts[-3]  # '11'
    last_part = parts[-1]        # '1877537' 或带有扩展名的文件名
    
    # 如果最后一部分包含文件扩展名，则去掉扩展名
    file_name, file_extension = os.path.splitext(last_part)
    
    # 构造新的文件名
    new_file_name = f"{third_last_part}+{file_name}"
    
    return new_file_name
teeth_dict = {
    'Incisors': { 'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0,'total_emd':0},
    'Canines': { 'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0,'total_emd':0},
    'Premolars': { 'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0,'total_emd':0},
    'Molars': {'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0,'total_emd':0}
}

def fidelity(pred_points,gt_points):
    assign_index_x2y = knn(pred_points,gt_points,1,batch_size=1)
    gt_to_pred_dist = torch.sqrt(torch.sum((gt_points - pred_points[assign_index_x2y[1]])**2, dim=1)) 
    return torch.mean(gt_to_pred_dist)
def update_teeth_statistics(teeth_dict, file_name, chamfer_dist,fidelity,f_value,val_emd):
    # 提取文件名中的象限编号（假设文件名以象限编号结尾）
   
  
    subdir = file_name.split('/')[-3]  # 调整此行以匹配实际文件命名格式
    # 根据象限编号确定牙齿类型
    if subdir.endswith(('1', '2')):
        tooth_type = 'Incisors'
    elif subdir.endswith('3'):
        tooth_type = 'Canines'
    elif subdir.endswith(('4', '5')):
        tooth_type = 'Premolars'
    elif subdir.endswith(('6', '7')):
        tooth_type = 'Molars'
    else:
        raise ValueError(f"Unrecognized subdir format: {subdir}")

    # 更新对应的牙齿类型的统计数据
    teeth_dict[tooth_type]['count'] += 1
    teeth_dict[tooth_type]['total_chamfer_distance'] += chamfer_dist
    teeth_dict[tooth_type]['total_fidelity'] += fidelity
    teeth_dict[tooth_type]['total_f_value'] += f_value
    teeth_dict[tooth_type]['total_emd'] += val_emd
def denormalize_point_cloud(normalized_point_cloud, point_cloud_center, crop_size):
    """
    De-normalize a point cloud from normalized coordinates back to original scale and position.

    Args:
        normalized_point_cloud (np.ndarray): Normalized point cloud with shape (num_points, 3).
        crop_center (np.ndarray): Original crop center with shape (3,).
        crop_scale (np.ndarray): Crop scale with shape (3,).

    Returns:
        np.ndarray: De-normalized point cloud.
    """
    # Reverse the scale and translation
    crop_center = np.array([crop_size / 2, crop_size / 2, crop_size / 2], dtype=np.float32)
    point_cloud = normalized_point_cloud * crop_size - crop_center + point_cloud_center
    
    return point_cloud

def hausdorff_95(A, B):
    # 计算A到B的单向Hausdorff距离
    max_A = torch.max(torch.min(torch.cdist(A, B), dim=1)[0])
    
    # 计算B到A的单向Hausdorff距离
    max_B = torch.max(torch.min(torch.cdist(B, A), dim=1)[0])
    
    # 取两者的最大值作为最终Hausdorff距离
    return torch.max(max_A, max_B)
# def hausdorff_95(A, B):
#     # 计算A到B的距离
#     dist_A_to_B = torch.min(torch.cdist(A, B), dim=1)[0]
#     # 计算B到A的距离
#     dist_B_to_A = torch.min(torch.cdist(B, A), dim=1)[0]

#     # 合并两个距离数组
#     all_distances = torch.cat((dist_A_to_B, dist_B_to_A))

#     # 排序距离
#     all_distances_sorted, _ = torch.sort(all_distances)

#     # 计算要排除的最大5%的索引
#     cutoff_index = int(len(all_distances_sorted) * 1)

#     # 排除掉最大5%的距离
#     truncated_distances = all_distances_sorted[:cutoff_index]

#     # 返回剩余部分的Hausdorff距离，即最大距离
#     return truncated_distances.max()
def test(model, test_loader, save_path='./test_outputs'):
    model.eval()  # 切换到评估模式

    val_chamfer = 0.0
    val_fidelity = 0.0
    val_emd = 0.0
    test_f_value = 0.0
    chamfer_name_pairs = []
    with torch.no_grad():

        for batch_idx, (inputs, targets,crown_center,template,psr_grid, crown_curvature,_,file_dir) in enumerate(tqdm(test_loader)):
            inputs,targets,template_tensor = inputs.to(accelerator.device), targets.to(accelerator.device),template.to(accelerator.device)
            # 前向传播
        
            pred_psr,seed,coarse, dense_cloud = model(inputs, template_tensor)
            #pred_psr,coarse, dense_cloud = model(inputs, template_tensor)
            #print(torch.mean(pred_psr-psr_grid.to('cuda')))
            if neg121:
                dense_cloud = dense_cloud / 2 +0.5
                targets = targets /2 +0.5
            
            
            batch_size = pred_psr.shape[0]
            for i in range(batch_size):
                file_name = file_dir[i]
                chamfer_dist = chamfer_distance(dense_cloud[i][None,:]*10.0,targets[i][None,:]*10.0,point_reduction='mean')[0]
                #chamfer_dist = hausdorff_95(dense_cloud[i][None,:]*10.0,targets[i][None,:]*10.0)
                print(chamfer_dist)
                #print(chamfer_dist)
                fidelity_val = chamfer_distance(dense_cloud[i][None,:]*10.0,targets[i][None,:]*10.0,single_directional=True)[0]
                #fidelity_val = fidelity(dense_cloud[i]*10.0,targets[i]*10.0)
                #print(fidelity_val)
                f_value= compute_fscore(targets[i]*10.0,dense_cloud[i]*10.0)
                #emd_value = earth_mover_distance(dense_cloud[i][None,:]*10,targets[i][None,:]*10,transpose=False)
                #emd_value = chamfer_distance(dense_cloud[i][None,:]*10.0,targets[i][None,:]*10.0,point_reduction='mean')[0]
                emd_value = hausdorff_95(dense_cloud[i][None,:]*10.0,targets[i][None,:]*10.0)
                name = extract_and_rename(file_name)
                chamfer_name_pairs.append((chamfer_dist.item(), name))
                val_chamfer += chamfer_dist
                val_fidelity += fidelity_val
                test_f_value += f_value
                val_emd += emd_value
                update_teeth_statistics(teeth_dict, file_name, chamfer_dist,fidelity=fidelity_val,f_value=f_value,val_emd =emd_value)
                outputs_np = dense_cloud[i].cpu().numpy()
                targets_np = targets[i].cpu().numpy()
                
                #point_cloud_target = pv.PolyData(targets_np)
                
                
                # print(name)
                output_filename = os.path.join(save_path, f"{name}_pc.ply")
                #target_filename = os.path.join(save_path, f"GT_{name}.ply")
                
                #point_cloud_target.save(target_filename)
                pred_psr_np = pred_psr[i]
                v, f, _ = mc_from_psr(pred_psr_np, zero_level=0)
                #pred_point_cloud = v*20.0 + crown_center[i].cpu().numpy() - np.array((10,10,10))
                pred_point_cloud = v*10.0 + crown_center[i].cpu().numpy()
                point_cloud_output = pv.PolyData(pred_point_cloud)
                #point_cloud_output.save(output_filename)
                # name = extract_and_rename(file_name)
                mesh_save_path = os.path.join(save_path,f"{name}.ply")
            
                #export_mesh(mesh_save_path,pred_point_cloud,f)
               
                # point_center = crown_centor[i].numpy()
                #file_name = file_dir[i]
                # if chamfer_dist>5:
                #     pred_psr_np = pred_psr[i]
                    # v, f, _ = mc_from_psr(pred_psr_np, zero_level=0)
                    # pred_point_cloud = v*20.0
                #     #pred_point_cloud = denormalize_point_cloud(v,point_center,25.6)
                   
                #     outputs_np = dense_cloud[i].cpu().numpy()
                #     template_np = template_tensor[i].cpu().numpy()
                #     print(outputs_np.shape)
                #     targets_np = targets[i].cpu().numpy()
                #     point_cloud_output = pv.PolyData(outputs_np)
                #     point_cloud_target = pv.PolyData(targets_np)
                #     template = pv.PolyData(template_np)
                #     output_filename = os.path.join(save_path, f"output_batch{batch_idx+1}_sample{i+1}.ply")
                #     target_filename = os.path.join(save_path, f"target_batch{batch_idx+1}_sample{i+1}.ply")
                #     template_filename = os.path.join(save_path, f"remplate_batch{batch_idx+1}_sample{i+1}.ply")
                #     point_cloud_output.save(output_filename)
                #     point_cloud_target.save(target_filename)
                #     template.save(template_filename)
                #     exit()
            # hausdorff = chamfer_distance(25.6*dense_cloud,25.6*targets,point_reduction="max")[0]
            # print(hausdorff)            # 累积验证损
            # val_hausdorff += hausdorff
           
            # # 保存前 4 个批次的输出为 .ply 文件
            # if batch_idx < 4:
            #     outputs_np = dense_cloud.cpu().numpy()  # 假设outputs是 (batch_size, 1, D, H, W)
            #     targets_np = targets.cpu().numpy()
            #     for i in range(outputs_np.shape[0]):
            #         # 转换为 PyVista 格式
            #         point_cloud_output = pv.PolyData(outputs_np[i])
            #         point_cloud_target = pv.PolyData(targets_np[i])

            #         # 保存为 .ply 文件
            #         output_filename = os.path.join(save_path, f"output_batch{batch_idx+1}_sample{i+1}.ply")
            #         target_filename = os.path.join(save_path, f"target_batch{batch_idx+1}_sample{i+1}.ply")
            #         point_cloud_output.save(output_filename)
            #         point_cloud_target.save(target_filename)

        # 返回验证集的平均 Hausdorff 距离
        val_chamfer /= len(test_loader.dataset)
        chamfer_name_pairs.sort(key=lambda x: x[0])
        with open(os.path.join(save_path, "chamfer_sorted.txt"), 'w') as f:
            for dist, name in chamfer_name_pairs:
                f.write(f"{name} {dist:.6f}\n")
        print(test_f_value/len(test_loader.dataset))
        print(val_fidelity/len(test_loader.dataset))
        print(teeth_dict)

    return val_chamfer



# 数据加载部分（测试集）
def load_test_data(batch_size=4, test_path='./test_data'):
    test_dataset = IOS_Datasetply4(test_path, is_train=False,sample_points=1600)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    return test_loader


# 测试主函数
def test_main():
    # 使用 argparse 传递超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--test_path', type=str, default='./test_data', help='Path to test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--save_path', type=str, default='./test_outputs/crowndeformer_6400_final_allteeth', help='Path to save the test results')
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)  # 如果不存在，创建目录
    # 模型初始化
    #model = PCN(num_coarse=512,detailed_output=True,grid_size=2)
    model = Pattn().to('cuda')

    # 加载训练好的模型
    ckpt = torch.load(args.model_path)
    ckpt = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(accelerator.device)

    # 加载测试数据
    test_loader = load_test_data(batch_size=args.batch_size, test_path=args.test_path)

    # 测试模型
    test_hausdorff = test(model, test_loader, save_path=args.save_path)
    print(f"Test hausdorff Coefficient:{test_hausdorff:.4f}")
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything(42)
    test_main()
