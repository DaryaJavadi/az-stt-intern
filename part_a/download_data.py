import os
import tarfile
import shutil
from dotenv import load_dotenv
from datacollective import download_dataset


load_dotenv()

api_key = os.getenv("MDC_API_KEY")
os.environ["MDC_API_KEY"] = api_key


# Downloading and Extracting Dataset:
dataset_id = "cmn29hqvk015ko107fblsr5ay"

dataset_path = download_dataset(dataset_id)
print("Downloaded to:", dataset_path)

downloaded_file = dataset_path

extract_path = "az_dataset"

os.makedirs(extract_path, exist_ok=True)

with tarfile.open(downloaded_file, "r:gz") as tar:
    tar.extractall(path=extract_path)

print("Extracted to:", extract_path)


# Clear structure:
BASE = "az_dataset/cv-corpus-25.0-2026-03-09/az"
NEW_BASE = "az_dataset"

os.makedirs(NEW_BASE, exist_ok=True)

# 1. move clips
shutil.copytree(
    f"{BASE}/clips",
    f"{NEW_BASE}/clips",
    dirs_exist_ok=True
)

# 2. move TSV files
for file in os.listdir(BASE):
    if file.endswith(".tsv"):
        shutil.copy(
            os.path.join(BASE, file),
            os.path.join(NEW_BASE, file)
        )

# 3. remove old nested folder
shutil.rmtree(
    os.path.join(
        extract_path,
        "cv-corpus-25.0-2026-03-09"
    )
)

print("Old nested folder removed")
print("Clean Structure Created")
