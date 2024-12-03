from chroma import Chroma, conditioners, api
import torch
import readline
import os
import time
from datetime import datetime

def save_pdb(filename, xyz):
    with open(filename, 'w') as f:
        for atomi, coord in enumerate(xyz[0]):
            x, y, z = coord
            atom_line = (f"ATOM  {atomi+1:5d}  CA  GLY A{atomi+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      C\n")
            f.write(atom_line)
        f.write("END\n")

def main():
    while True:
        print("\nEnter new input VOXEL path (or 'exit' to quit):")
        voxel_path = input("> ").strip()
        
        if voxel_path.lower() == 'exit':
            print("Exiting program...")
            break
        
        if os.path.exists(voxel_path):
            print("\nEnter new length of a protein to generate [default=auto]:")
            length = int(input("> ").strip() or 0)
            print("auto" if length==0 else length)
            
            print("\nEnter new step size to parse a voxel.")
            print("It is highly related to a resolution and 5A is enough [default=2]:")
            step = int(input("> ").strip() or 2)
            print(step)
            
            print("\nEnter new SDF threshold value to generate a target shape [default=0]:")
            threshold = int(input("> ").strip() or 0)
            print(threshold)
            
            print('\nSetting a conditioner')
            start_time = time.time()
            conditioner = conditioners.SDFConditioner(
                voxel_path,
                chroma.backbone_network.noise_schedule,
                autoscale=False,
                autolength=length==0,
                autoscale_num_residues=length,
                step=step,
                threshold=threshold,
            ).to(device)
            conditioner_time = time.time() - start_time
            print(f'Conditioner setup time: {conditioner_time:.2f} seconds')
            
            if length==0:
                print(f'\nRecommended amino acid length range is [{conditioner.autolength_min}, {conditioner.autolength_max}].')
                print(f'{conditioner.autoscale_num_residues} was chosen from the range.')
            
            print('\nGenerating a structure')
            start_time = time.time()
            shaped_protein, trajectories = chroma.sample(
                chain_lengths=[conditioner.autoscale_num_residues], conditioner=conditioner, full_output=True, samples=1, steps=500
            )
            generation_time = time.time() - start_time
            print(f'Structure generation time: {generation_time:.2f} seconds')
            
            time_suffix = datetime.now().strftime("%m%d_%H%M%S")
            out_path = voxel_path.replace('.voxel',f'_{time_suffix}.pdb')
            shaped_protein.to(out_path)
            print(f'\nSaved the output to\n> {out_path}')
            
            out_path = voxel_path.replace('.voxel','_X_target.pdb')
            save_pdb(out_path, conditioner.X_target)
            print(f"Saved the target shape to\n> {out_path}")
            
            sdf_loss = conditioner.sdfs[-1]
            if sdf_loss < 1:
                emoji = "😘"
            elif sdf_loss < 2:
                emoji = "🙂"
            elif sdf_loss < 3:
                emoji = "😶"
            else:
                emoji = "🙁"
                
            print(f'\nSDF loss:{sdf_loss:.2f} {emoji}')
            print(f'D_w loss:{conditioner.D_ws[-1]:.2f}')
            
        
        else:
            print(f"Could not found '{voxel_path}', try again.")
    return

if __name__=='__main__':
    api.register_key("b287b5c091514189ab14cf0b55b795bb")
    device="cuda" if torch.cuda.is_available() else 'cpu'
    print(f'Device is {device}')
    
    print('Loading weights')
    start_time = time.time()
    chroma = Chroma(
        weights_backbone='/home/annung202/chroma/assets/chroma_weights/chroma_backbone_v1.0.pt', 
        weights_design='/home/annung202/chroma/assets/chroma_weights/chroma_design_v1.0.pt'
    ).to(device)
    loading_time = time.time() - start_time
    print(f'Weights loading time: {loading_time:.2f} seconds')
    
    main()