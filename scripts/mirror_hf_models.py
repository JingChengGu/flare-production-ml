from pathlib import Path
from huggingface_hub import snapshot_download, create_repo, upload_folder

models = {
    "irvingz/segformer-b3-finetuned-segments-chargers-full-v3.1": "JaesonGu/flare-segformer-mit-b3",
    "dskong07/screen-classif-model": "JaesonGu/flare-screen-vit",
    "dskong07/charger-classif-model": "JaesonGu/flare-body-vit",
    "dskong07/cord-classif-model": "JaesonGu/flare-cable-vit",
    "dskong07/plug-classif-model": "JaesonGu/flare-plug-vit",
}

local_base_dir = Path("./hf_model_mirror_cache")
local_base_dir.mkdir(exist_ok=True)

for source_repo, target_repo in models.items():
    print(f"\nDownloading {source_repo}...")

    local_dir = local_base_dir / source_repo.replace("/", "__")

    snapshot_download(
        repo_id=source_repo,
        repo_type="model",
        local_dir=str(local_dir),
    )

    print(f"Creating target repo {target_repo}...")

    create_repo(
        repo_id=target_repo,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    print(f"Uploading files to {target_repo}...")

    upload_folder(
        folder_path=str(local_dir),
        repo_id=target_repo,
        repo_type="model",
        commit_message=f"Mirror model from {source_repo}",
    )

    print(f"Finished mirroring {source_repo} → {target_repo}")

print("\nAll requested models mirrored successfully.")