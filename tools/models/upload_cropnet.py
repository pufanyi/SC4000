from huggingface_hub import HfApi


api = HfApi()

model_path = (
    "/data/pufanyi/project/SC4000/output/models/MobileNetV3_2024-10-27-22-54-49/model"
)
repo_id = "pufanyi/SC4000-MobileNetV3"

api.create_repo(repo_id, repo_type="model")

api.upload_folder(
    folder_path=model_path,
    path_in_repo="/",
    repo_id=repo_id,
    repo_type="model",
)
