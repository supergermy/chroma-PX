from chroma import Chroma, conditioners, api
import torch
import readline
import os
import time
from datetime import datetime

def save_pdb_X_target(filename, xyz):
    with open(filename, 'w') as f:
        for atomi, coord in enumerate(xyz[0]):
            x, y, z = coord
            atom_line = (f"ATOM  {atomi+1:5d}  CA  GLY A{atomi+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00            C\n")
            f.write(atom_line)
        f.write("END\n")

def save_pdb_X_target_SO3(filename, xyz):
    with open(filename, 'w') as f:
        for atomi, coord in enumerate(xyz[0]):
            x, y, z = coord
            atom_line = (f"ATOM  {atomi+1:5d}  CA  GLY A{atomi+1:4d}    "
                        f"{z:8.3f}{y:8.3f}{-x:8.3f}  1.00  0.00            C\n")
            f.write(atom_line)
        f.write("END\n")

# def save_pdb_X_target_SE3(filename, xyz, R3):
#     with open(filename, 'w') as f:
#         for atomi, coord in enumerate(xyz[0]):
#             x, y, z = coord
#             atom_line = (f"ATOM  {atomi+1:5d}  CA  GLY A{atomi+1:4d}    "
#                         f"{z+R3[0,0]:8.3f}{y+R3[0,1]:8.3f}{-x+R3[0,2]:8.3f}  1.00  0.00            C\n")
#             f.write(atom_line)
#         f.write("END\n")

# def save_pdb(filename, xyz):
#     atomn = ['N','CA','C','O']
#     with open(filename, 'w') as f:
#         for resi, coords in enumerate(xyz[0]):
#             for _atomi, xyz in enumerate(coords):
#                 x, y, z = xyz
#                 atom_line = (f"ATOM  {resi*4+_atomi+1:5d}  {atomn[_atomi]:<2}  GLY A{resi+1:4d}    "
#                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atomn[_atomi][0]} \n")
#                 f.write(atom_line)
#         f.write("END\n")
        
def save_pdb_SO3(filename, xyz):
    atomn = ['N','CA','C','O']
    with open(filename, 'w') as f:
        for resi, coords in enumerate(xyz[0]):
            for _atomi, xyz in enumerate(coords):
                x, y, z = xyz
                atom_line = (f"ATOM  {resi*4+_atomi+1:5d}  {atomn[_atomi]:<2}  GLY A{resi+1:4d}    "
                            f"{z:8.3f}{y:8.3f}{-x:8.3f}  1.00  0.00           {atomn[_atomi][0]} \n")
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
            
            # Why do we even need this?
            # print("\nEnter new SDF threshold value to generate a target shape [default=0]:")
            # threshold = int(input("> ").strip() or 0)
            # print(threshold)
            
            print('\nSetting a conditioner')
            start_time = time.time()
            conditioner = conditioners.SDFConditioner(
                voxel_path,
                chroma.backbone_network.noise_schedule,
                autoscale=False,
                autolength=length==0,
                autoscale_num_residues=length,
                step=step,
                # threshold=threshold,
            ).to(device)
            conditioner_time = time.time() - start_time
            print(f'Conditioner setup time: {conditioner_time:.2f} seconds')
            
            if length==0:
                print(f'\nRecommended amino acid length range is [{conditioner.autolength_min}, {conditioner.autolength_max}].')
                print(f'{conditioner.autoscale_num_residues} was chosen from the range.')
            
            # out_path = voxel_path.replace('.voxel','_X_target_O.pdb')
            # save_pdb_X_target(out_path, conditioner.X_target.cpu())
            # print(f"Saved the target shape to\n> {out_path}")
            
            out_path = voxel_path.replace('.voxel','_X_target_original.pdb')
            save_pdb_X_target(out_path, conditioner.voxel_grid.X_target_original.cpu().view([1,-1,3]))
            print(f"Saved the target shape to\n> {out_path}")
            
            out_path = voxel_path.replace('.voxel','_X_target_O_R.pdb')
            save_pdb_X_target_SO3(out_path, conditioner.X_target.cpu())
            print(f"Saved the target shape to\n> {out_path}")
            
            # out_path = voxel_path.replace('.voxel','_X_target_O_RT.pdb')
            # save_pdb_X_target_SE3(out_path, conditioner.X_target.cpu(), conditioner.voxel_grid.X_target_center)
            # print(f"Saved the target shape to\n> {out_path}")
            
            # out_path = voxel_path.replace('.voxel','_X_target.pdb')
            # save_pdb_X_target(out_path, conditioner.X_target.cpu()+conditioner.voxel_grid.X_target_center)
            # print(f"Saved the target shape to\n> {out_path}")
            
            print('\nGenerating a structure')
            start_time = time.time()
            shaped_protein, trajectories = chroma.sample(
                chain_lengths=[conditioner.autoscale_num_residues], conditioner=conditioner, full_output=True, samples=1, steps=500
            )
            generation_time = time.time() - start_time
            print(f'Structure generation time: {generation_time:.2f} seconds')
            
            time_suffix = datetime.now().strftime("%m%d_%H%M%S")
            # out_path = voxel_path.replace('.voxel',f'_{time_suffix}_O.pdb')
            # shaped_protein.to(out_path)
            # print(f'\nSaved the centered output to\n> {out_path}')
            
            out_path = voxel_path.replace('.voxel',f'_{time_suffix}_O_R.pdb')
            save_pdb_SO3(out_path, shaped_protein.to_XCS()[0].cpu())
            print(f"Saved the target shape to\n> {out_path}")
            
            # out_path = voxel_path.replace('.voxel',f'_{time_suffix}.pdb')
            # save_pdb(out_path,shaped_protein.to_XCS()[0].cpu()+conditioner.voxel_grid.X_target_center)
            # print(f'\nSaved the output to\n> {out_path}')
            
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