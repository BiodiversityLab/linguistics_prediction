import os
import glob
import yaml
import shutil


def delete_runs_with_deleted_lifecycle(artifact_uri):
    # Hämta alla mappar under artifact_uri
    run_folders = glob.glob(os.path.join(artifact_uri, "*"))

    for folder in run_folders:
        meta_file = os.path.join(folder, "meta.yaml")
        if os.path.exists(meta_file):
        
            removerun = False
            with open(meta_file, "r") as f:
                meta_data = yaml.safe_load(f)
                lifecycle_stage = meta_data.get("lifecycle_stage")

                if lifecycle_stage == "deleted":
                    removerun = True
            if removerun:
                # Ta bort hela mappen för den här körningen
                print(f"Tar bort körning med run_id: {meta_data['run_id']}")
                
                #os.removedirs(folder)
                shutil.rmtree(folder)

# Använd din specifika artifact_uri här
artifact_uri = "C:/biodiveristy/cnn_diversity_modelling/mlruns/803114424460668909/"
delete_runs_with_deleted_lifecycle(artifact_uri)
