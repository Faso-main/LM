from huggingface_hub import HfApi, Repository, notebook_login # pip install huggingface_hub, ipywidgets
import os, json
from datetime import datetime
import pandas as pd
# дополнительно нужно пройти авторизацию по ключу(>> huggingface-cli login), за ключом в личку


"""
Добавить стороки ниже в ключевой if __name__ после main()
HF_REPO_NAME = "faso312/test1"  # Прописал свое имя, свой аккаунт, возможно стоит сделать под это аккаунт команды
upload_to_huggingface(SAVE_PATH, HF_REPO_NAME, RESULTS_PATH, IMG_PATH, label_path)
"""

# Добавьте эту функцию в ваш код
def upload_to_huggingface(model_path, repo_name, results_path, img_path, label_path , epochs):
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
        path_or_fileobj=label_path,
        path_in_repo="label2id.json",
        repo_id=repo_name,
        repo_type="model"
    )
    readme_content = f"""
# {repo_name.split('/')[-1]}

This is a model repository for {repo_name.split('/')[-1]}.

## Model Details
[Тренировка масочных моделей для проекта "Анализ неструктурированных медицинских данных]

## Training
This model was trained with [PyTorch, Transformers].
[Количество эпох {epochs}].

## Results
Валидацию после тренировки храниться в файле results.csv

## Usage
 - [Пример кода для использования вашей модели, ссылка на гитхаб(Faso-main/LM)](https://github.com/Faso-main/LM/blob/main/NLP_Med/src/hf/HF_download.py)

## Files

  - `model_path/`: Contains the model weights and configuration.

  - `results.csv`: Contains training and evaluation results.

  - `training_plot.png`: Plot of training metrics.

  - `label2id.json`: Mapping from labels to IDs.
"""

    temp_readme_path = os.path.join('NLP_Med','src','temp_README.md')
    with open(temp_readme_path, "w", encoding="utf-8") as temp: temp.write(readme_content)
    api.upload_file(
        path_or_fileobj=temp_readme_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model"
    )
    os.remove(temp_readme_path) # удаляем временный файл
    print(f"All files uploaded to https://huggingface.co/{repo_name}")