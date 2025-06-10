from huggingface_hub import HfApi, Repository, notebook_login # pip install huggingface_hub, ipywidgets
# дополнительно нужно пройти авторизацию по ключу(>> huggingface-cli login), за ключом в личку

# Добавьте эту функцию в ваш код
def upload_to_huggingface(model_path, repo_name, results_path, img_path, lable_path):
    # Аутентификация
    notebook_login()
    
    # Создаем репозиторий
    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)
    
    # Загружаем модель
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model"
    )
    
    # Загружаем результаты
    api.upload_file(
        path_or_fileobj=results_path,
        path_in_repo="results.csv",
        repo_id=repo_name,
        repo_type="model"
    )
    
    # Загружаем графики
    api.upload_file(
        path_or_fileobj=img_path,
        path_in_repo="training_plot.png",
        repo_id=repo_name,
        repo_type="model"
    )
    
    # Загружаем label2id.json
    api.upload_file(
        path_or_fileobj=lable_path,
        path_in_repo="label2id.json",
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"All files uploaded to https://huggingface.co/{repo_name}")