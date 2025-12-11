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
    def __init__(self, root_path, rgb_path, depth_path, device=None):

        self.rgb_path = os.path.join(root_path, rgb_path)
        self.depth_path = os.path.join(root_path, depth_path)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if not os.path.exists(self.depth_path):
            raise FileNotFoundError(f"Depth file not found: {self.depth_path}")

        # Parse Header
        with open(self.depth_path, "rb") as f:
            header_data = f.read(20)
            unpacked = struct.unpack("IIiff", header_data)
            self.width = unpacked[0]
            self.height = unpacked[1]
            self.half_fov_x = unpacked[3]
            self.half_fov_y = unpacked[4]
            
            f.seek(0, os.SEEK_END)
            total_bytes = f.tell()
            data_bytes_per_frame = self.width * self.height * 2 
            stride = 4 + data_bytes_per_frame
            self.num_frames = (total_bytes - 20) // stride

        print(f"[Loader] {self.width}x{self.height} | {self.num_frames} Frames")
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
        if frame_idx >= self.num_frames:
            frame_idx = 0

        # Read RGB
        rgb_stride = 4 + self.width * self.height * 4
        rgb_offset = 20 + frame_idx * rgb_stride + 4 
        
        with open(self.rgb_path, "rb") as f:
            f.seek(rgb_offset)
            raw_rgb = np.fromfile(f, dtype=np.uint8, count=self.width * self.height * 4)
            # Normalize to 0-1
            rgb = torch.from_numpy(raw_rgb.reshape(-1, 4).astype(np.float32) / 255.0).to(self.device)

        # Read Depth
        depth_stride = 4 + self.width * self.height * 2
        depth_offset = 20 + frame_idx * depth_stride + 4 
        
        with open(self.depth_path, "rb") as f:
            f.seek(depth_offset)
            raw_depth = np.fromfile(f, dtype=np.uint16, count=self.width * self.height)
            depth_meters = torch.from_numpy(raw_depth.astype(np.float32) / 1000.0).to(self.device)

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
        
        scale = 1.0 / self.tan_fov_x
        points = torch.stack([x * scale, y * scale, z * scale], dim=-1)
        return points

    def get_pointcloud(self, frame_idx):
        rgb, depth = self.get_frame(frame_idx)
        points = self.unproject(depth)
        
        # Filter Background (> 3m) and invalid depth (< 0.1m)
        mask = (depth > 0.1) & (depth < 3.5)
        
        valid_points = points[mask]
        valid_rgb = rgb[mask]

        # BGRA -> RGBA
        valid_rgb = valid_rgb[:, [2, 1, 0]] 
        
        return Pointclouds(points=[valid_points], features=[valid_rgb])


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("../data/dataset.json"):
    print("dataset.json not found")
    exit()

with open("../data/dataset.json", 'r') as f:
    config = json.load(f)

cam_config = config['cameras'][0]
print(f"Processing: {cam_config['id']}")

loader = TemPCCStreamLoader("../data", cam_config['rgb_path'], cam_config['depth_path'], device=device)

# Initialize Polyscope
ps.init()
ps.set_up_dir("y_up") # Adjust if your camera is rotated
ps.set_ground_plane_mode("shadow_only")

# State for the animation loop
current_frame = 0
is_playing = True
# ps_cloud.set_radius(0.005) 



def update_loop():
    global current_frame, is_playing

    # GUI Control to Pause/Play
    if ps_imgui.Button("Pause" if is_playing else "Play"):
        is_playing = not is_playing
    
    # Slider to scrub through video
    changed, new_frame = ps_imgui.SliderInt("Frame", current_frame, 0, loader.num_frames - 1)
    if changed:
        current_frame = new_frame
        is_playing = False # Pause if manually scrubbing
  

    # Get data from your existing loader
    pc = loader.get_pointcloud(frame_idx=current_frame)
    
    # Extract raw tensors from PyTorch3D structure
    # Polyscope needs raw (N,3) arrays, not Pointclouds objects
    if len(pc) > 0:
        verts = pc.points_padded()[0]
        colors = pc.features_padded()[0]

        # Register with Polyscope
        # Converting to CPU numpy is safest for Polyscope compatibility
        ps_cloud = ps.register_point_cloud("TemPCC Stream", verts.cpu().numpy())
        ps_cloud.add_color_quantity("RGB", colors.cpu().numpy(), enabled=True)
        
        # ps_cloud.set_radius(0.005) 

    # Advance frame
    if is_playing:
        current_frame = (current_frame + 1) % loader.num_frames

# Register callback and start
ps.set_user_callback(update_loop)
ps.show()