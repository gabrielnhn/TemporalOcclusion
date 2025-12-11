import os
import json
import struct
import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as ps_imgui
from pytorch3d.structures import Pointclouds

import os.path

class TemPCCStreamLoader:
    def __init__(self, root_path, rgb_path, depth_path, transformation_matrix, device=None):
        self.rgb_path = os.path.join(root_path, rgb_path)
        self.depth_path = os.path.join(root_path, depth_path)
        self.id = rgb_path # Just for debugging
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if not os.path.exists(self.depth_path):
            raise FileNotFoundError(f"Depth file not found: {self.depth_path}")

        with open(self.depth_path, "rb") as f:
            header_data = f.read(20)
            unpacked = struct.unpack("IIiff", header_data)
            self.width = unpacked[0]
            self.height = unpacked[1]
            self.half_fov_x = unpacked[3]
            self.half_fov_y = unpacked[4]
            
            f.seek(0, os.SEEK_END)
            total_depth_bytes = f.tell()
            depth_stride = 4 + self.width * self.height * 2 
            num_depth_frames = (total_depth_bytes - 20) // depth_stride

        with open(self.rgb_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            total_rgb_bytes = f.tell()
            rgb_stride = 4 + self.width * self.height * 4
            num_rgb_frames = (total_rgb_bytes - 20) // rgb_stride

        self.num_frames = min(num_depth_frames, num_rgb_frames)
        
        if self.num_frames <= 0:
            raise ValueError(f"Camera has 0 valid frames (Depth: {num_depth_frames}, RGB: {num_rgb_frames})")

        self.matrix = torch.tensor(transformation_matrix, device=self.device).reshape([4,4]).T



        print(f"[Loader] Loaded {self.num_frames} frames ({self.width}x{self.height})")
        self._precompute_grid()

    def _precompute_grid(self):
        y_indices, x_indices = torch.meshgrid(
            torch.arange(self.height, device=self.device), 
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )
        self.grid_x = x_indices.flatten().float()
        self.grid_y = y_indices.flatten().float()
        self.tan_fov_x = np.tan(self.half_fov_x)
        self.tan_fov_y = np.tan(self.half_fov_y)

    def get_frame(self, frame_idx):
        # Safety wrap
        if frame_idx >= self.num_frames:
            frame_idx = 0

        rgb_stride = 4 + self.width * self.height * 4
        rgb_offset = 20 + frame_idx * rgb_stride + 4 
        expected_rgb_count = self.width * self.height * 4
        
        with open(self.rgb_path, "rb") as f:
            f.seek(rgb_offset)
            raw_rgb = np.fromfile(f, dtype=np.uint8, count=expected_rgb_count)
            
        if raw_rgb.size != expected_rgb_count:
            raw_rgb = np.zeros(expected_rgb_count, dtype=np.uint8)

        rgb = torch.from_numpy(
            raw_rgb.reshape(-1, 4).astype(np.float32) / 255.0
        ).to(self.device)

        depth_stride = 4 + self.width * self.height * 2
        depth_offset = 20 + frame_idx * depth_stride + 4 
        
        with open(self.depth_path, "rb") as f:
            f.seek(depth_offset)
            raw_depth = np.fromfile(f, dtype=np.uint16, count=self.width * self.height)
            
            if raw_depth.size != self.width * self.height:
                 raw_depth = np.zeros(self.width * self.height, dtype=np.uint16)

        depth_meters = torch.from_numpy(
            raw_depth.astype(np.float32) / 1000.0
        ).to(self.device)

        return rgb, depth_meters

    def unproject(self, depth_map):
        sX = self.grid_x
        sY = self.grid_y
        sX_inv = self.width - sX - 1 

        term_x = -((sX_inv * 2.0 / self.width) - 1.0)
        term_y = -((sY * 2.0 / self.height) - 1.0)
        
        z = depth_map
        x = self.tan_fov_x * term_x * z
        y = self.tan_fov_y * term_y * z
        
        # DONT APPLY?
        # points = torch.stack([x, y, z], dim=-1)

        # APPLY?
        points_homogeneous = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)
        transform = self.matrix.to(points_homogeneous.device).type(points_homogeneous.dtype)
        mult_result = torch.mm(transform, points_homogeneous.T).T
        homogeneous_denom = mult_result[:, 3].unsqueeze(1)
        homogeneous_denom[homogeneous_denom == 0] = 1e-6
        points = mult_result[:, :3] / homogeneous_denom

        return points

    def get_pointcloud(self, frame_idx, maxdepth=3.0):
        rgb, depth = self.get_frame(frame_idx)
        points = self.unproject(depth)
        
        # Filter Background (> 3m?) and invalid depth (< 0.1m)
        mask = (depth > 0.001) & (depth < maxdepth) & (torch.any(rgb, dim=1) > 0.3)

        # print(mask.shape)
        # mask = (depth > 0.1)
        
        valid_points = points[mask]
        valid_rgb = rgb[mask]

        # BGRA -> RGBA
        valid_rgb = valid_rgb[:, [2, 1, 0]] 

        # return Pointclouds(points=[valid_points], features=[valid_rgb])
        return valid_points, valid_rgb

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("../data2/dataset.json"):
    print("dataset.json not found")
    exit()

with open("../data2/dataset.json", 'r') as f:
    config = json.load(f)

loaders = []

for cam_idx in range(len(config['cameras'])):
    cam_config = config['cameras'][cam_idx]
    print(f"Processing: {cam_config['id']}")
    loader = TemPCCStreamLoader(
        "../data2",
        cam_config['rgb_path'],
        cam_config['depth_path'],
        cam_config['transformation_matrix'],
        device=device
    )
    loaders.append(loader)


# Initialize Polyscope
ps.init()
ps.set_up_dir("y_up") 
ps.set_ground_plane_mode("shadow_only")

# State for the animation loop
current_frame = 0
is_playing = True
maxdepth = 3.0

cameras_checkboxes = [True for loader in loaders]

def update_loop():
    global current_frame, is_playing, maxdepth

    
    for idx, checkbox in enumerate(cameras_checkboxes):
        changed, newvalue = ps_imgui.Checkbox(f"Camera{idx}", checkbox)
        cameras_checkboxes[idx] = newvalue
    
    # GUI Control to Pause/Play
    if ps_imgui.Button("Pause" if is_playing else "Play"):
        is_playing = not is_playing
    
    # Slider to scrub through video
    changed, new_frame = ps_imgui.SliderInt("Frame", current_frame, 0, loaders[0].num_frames - 1)
    if changed:
        current_frame = new_frame
        is_playing = False

    changed, newmaxdepth = ps_imgui.SliderFloat("Max Depth", maxdepth, v_min=0.1, v_max=10)
    if changed:
        maxdepth = newmaxdepth

    all_verts = []
    all_colors = []

    for i, loader in enumerate(loaders):
        if not cameras_checkboxes[i]:
            continue

        verts, colors = loader.get_pointcloud(frame_idx=current_frame, maxdepth=maxdepth)
        
        all_verts.append(verts)
        all_colors.append(colors)

    if all_verts:
        combined_verts = torch.cat(all_verts, dim=0)
        combined_colors = torch.cat(all_colors, dim=0)

        # Register combined cloud
        ps_cloud = ps.register_point_cloud("Full Scene", combined_verts.cpu().numpy())
        ps_cloud.add_color_quantity("RGB", combined_colors.cpu().numpy(), enabled=True)
        # ps_cloud.set_radius(0.005)

    # Advance frame
    if is_playing:
        current_frame = (current_frame + 1) % loader.num_frames

# Register callback and start
ps.set_max_fps(30)
ps.set_user_callback(update_loop)
ps.show()