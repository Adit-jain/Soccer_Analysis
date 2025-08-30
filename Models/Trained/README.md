By Default, this is the directory for all the newly trained. The fine tuned models will be stored here.

## Pre-trained Models

### Object Detection Model
Download the soccer analysis object detection model:

```bash
# Using huggingface_hub
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Adit-jain/soccana", filename="best.pt", local_dir="./Models/Trained/")
```

```bash
# Using git (requires git-lfs)
git clone https://huggingface.co/Adit-jain/soccana
```

### Keypoint Detection Model
Download the soccer field keypoint detection model:

```bash
# Using huggingface_hub
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Adit-jain/Soccana_Keypoint", filename="best.pt", local_dir="./Models/Trained/")
```

```bash
# Using git (requires git-lfs)
git clone https://huggingface.co/Adit-jain/Soccana_Keypoint
```