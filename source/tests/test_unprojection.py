import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)

import omni.isaac.core.utils.prims as prim_utils
import os
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth, create_semantic_pointcloud_from_depth_and_seg
import torch

def test_unprojection():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_robots = 4
    
    intrinsic_matrix = torch.tensor([[732.9993,   0.0000, 320.0000],
                                     [  0.0000, 732.9993, 240.0000],
                                     [  0.0000,   0.0000,   1.0000]], device=device)
    pos_w = torch.tensor([ 6.5844, -4.9575,  0.1335], device=device)
    quat_w_ros = torch.tensor([ 0.5010, -0.4982, -0.4990,  0.5018], device=device)
    
    intrinsic_matrix = intrinsic_matrix.unsqueeze(0).repeat(num_robots, 1, 1)
    pos_w = pos_w.unsqueeze(0).repeat(num_robots, 1)
    quat_w_ros = quat_w_ros.unsqueeze(0).repeat(num_robots, 1)
    
    semantic = torch.randint(0, 5, (num_robots, 480, 640), dtype=torch.int32)
    depth = torch.rand((num_robots, 480, 640), dtype=torch.float32)
    
    pc_labeled = create_semantic_pointcloud_from_depth_and_seg(
        intrinsic_matrix=intrinsic_matrix,
        depth=depth,
        semantic=semantic,
        position=pos_w,
        orientation=quat_w_ros,
        device=device,
        keep_invalid=True
    )
    
if __name__ == "__main__":
    test_unprojection()