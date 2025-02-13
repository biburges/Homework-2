# Homework-2 (Segment-Anything-2 by Meta)
## Brief Background on SAM-2
Image detection remains a primary focus of much of modern day robotics applications, given that it provides a medium to model , encode and extract our dynamic real world.
Existing at the forefront of these algorithims exists SAM-2, Meta's Segment Anything Model 2, an open source and free software that performs promptable image segmentation of images and videos.
Contrary to the status quo at the time, SAM-2 masking, pointing and bounding box applications enables users to define an area of interest within a frame (even if the software has never been trained on it before) and focus/extract on said subject even through challenges such as reapperance, collision (deformation) and image destabilization. 

SAM2 improves upon its predecessor, SAM1, by increased compatibility with images and videos, the supporting of real-time video segmentation and the achieving of image processing speeds upto 6x faster than previous models. As evidence below, complex shapes experience deformation, occulsion and reapperance are handled with extreme accuracy. This github repo aims to detail the features of SAM-2, explain its applications and guide users through the installation and running of the software.

![image](https://github.com/user-attachments/assets/117001c7-c328-4045-92f5-2243581209d7)
###### "Image from Towards AI: SAM2 is Amazing but we need to understand SAM1 by Jaigensan"

## Architecture
![image](https://github.com/user-attachments/assets/c01cfdba-eec0-4efb-b4c5-567efdde8245)
###### "Image from [github.com/facebookresearch/sam2](https://www.datacamp.com/blog/sam2-meta-segment-anything-model)"

### Components
- Image and Video Encoder
    - It is a hierarchical masked autoencoder which allows for the extraction of high-level features during decoding of images/videos. Thus, it encodes the video frames one frame at a time (per timestep)
- Memory Mechanism (memory encoder, memory bank, memory attention module)
    - A cross-attention mechanism storing and utilizing previous frames features, enabling predictions and prompts that manifest in consistent tracking over time.
- Prompt Encoder
    - Uses self-attention and cross-attention framework to handle different types of user defined prompts (points,boxes,masks). This enables focus on the segmentation of tasks. represented by positional encodings and learned embeddings and dense prompts(handled by convolutional layers).
- Mask decoder
    - The mask decoder uses the encoded prompts, and the encoded and conditioned frames from the memory attention module to generate the mask for the current frame This fast mask decoding from encoding allows for real time segmentation. This helps create high-resolution information for decoding

 ### Flow of information through architecture
 
The interaction between the image/video encoder, the prompt encoder and the mask decoder is as follows. As displayed in the image above, the image encoder extracts fine and broad patterns/features from the frame. This information is segmented using information from the prompt encoder and stored in the memory mechanism. This information is used to improve mask predictation for future frame. Finally the rapid processing of encoder information creates dynamic real time segmentation. 

### Create data engine with model in the loop

- Phase 1
   - First, SAM is used to generate masks and assist humans with annotations. This is then followed by human annotators that correct and refine the annotations on frames of videos  with pixel-precise editing tools.
- Phase 2
    - The mask from SAM 1 and manual masks are used as prompts for the SAM 2 model. Using the prompted masks from the first frame, subsequent masks are created by the SAM 2 model
- Phase 3
    - Only SAM 2 is used and accepts all types of prompts
 
### Benefits of using SAM2 over SAM1/Other segmentation software
- Minimal human intervention is needed due to lack of retraining of model, displayed by segmentation of new information
- Real time due to efficiency and speed of software
- Enables specific masking application due to teh acceptance of prompts for segmentation
- Open source and free

### Applications of SAM-2
- **Entertainment and Photography Industry:** Due to its ability to perform high speed/quality segmentation of dynamic scenes irrespective of occulsion and reapperance, SAM-2 remains a great software to assist in background removal, replacement as well as subtraction of elements from a scene. With the advent of AI, this could also manifest in the addition of elements and scenes further enriching the industry and lowering the barriers to entry
- **Traffic and Autonomous Driving:** The modeling of traffic and obstacle detection remain prominent challenges in the space of autonomous driving, especially given the sensitive nature of the industry (human lives at stake). Increasingly accurate detection of objects in addition to the prediction of how elements move through a dynamic setting could improve the safety of the autonomous vehicles whilst also helping law enforcement uphold traffic laws
- **Research and Medicine:** Given that underlying sicknesses and diseases often display themselves in the form of tumors, bumps, rashes etc. increasing usage of models like SAM-2 could improve the chances of detection and characterization of sicknesses, possibly contributing to increased recovery quality

## How to Install Code and Libraries(Best for Linux)
### Before Installation ###
Make sure you have the following:

**wsl**
- Installation code in Windows Powershell (run as admin):
  ```
  wsl --install
  ```

**Anaconda**
- Installation code:
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
- Verify the Version
    ```
    conda --version
    ```
**PyTorch**
- Installation code
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
**OpenCV**
- Installation code
    ```
    pip install opencv-contrib-python
    ```
**Jupyter Notebook**
- Installation code
    ```
    pip install notebook
    ```
**Matplotlib**
- Installation code
    ```
    pip install matplotlib
    ```
**Pillow**
-Installation code
    ```
    pip install Pillow
    ```
### During Installation of SAM2 ###
Enter Anaconda Environment
```
conda create -n sam2 python=3.12
```
Activate Anaconda
```
conda activate sam2
```
Clone the Repository
```
git clone https://github.com/facebookresearch/segment-anything-2.git
```
Next, Enter the Directory and install SAM 2
```	
cd segment-anything-2
```
```	
pip install -e .
```
Then, Install the Depedencies for the demo
```
pip install -e ".[demo]"

```
Then, Download all the Checkpoints

```
cd checkpoints
./download_ckpts.sh
```

## How to Run the Code(Using their stock images and videos)
**Setup**
Go to the notebooks folder through the directory and open the jupyter notebook titled "video_predictor_example.ipynb"

Open Colab using the link provided in the jupyter script(ignore if just using jupyter notebook)

Execute the code stating you are using Colab to run SAM2
```
using_colab = True
```
Import software
```
if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
    !{sys.executable} -m pip install opencv-python matplotlib
    !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/sam2.git'

    !mkdir -p videos
    !wget -P videos https://dl.fbaipublicfiles.com/segment_anything_2/assets/bedroom.zip
    !unzip -d videos videos/bedroom.zip

    !mkdir -p ../checkpoints/
    !wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```
```
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
```
Select the device for computation
```
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
**Loading the SAM 2 video predictor**
Setup predictor based on checkpoints
```
```
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
```
Display Mask and Points
```
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
```
Upload your sample image into SAM2(stock provided by SAM2)
```
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "./videos/bedroom"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
```
Initialize the Inference State
```
inference_state = predictor.init_state(video_path=video_dir)
```
ONLY USE TO RESET THE INFERENCE STATE

```
predictor.reset_state(inference_state)
```
First Positive Click on Frame
```
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[210, 350]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
```
Second Positive Click on Frame
```
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
```
Propogate the prompts to get the masklet across the video(may take a long time or cause kernel to die)
```
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
```
Further refine the masklet by creating new prompts to make negative clicks
```
ann_frame_idx = 150  # further refine some details on this frame
ann_obj_id = 1  # give a unique id to the object we interact with (it can be any integers)

# show the segment before further refinement
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx} -- before refinement")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_mask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=ann_obj_id)

# Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
points = np.array([[82, 410]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([0], np.int32)
_, _, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the segment after the further refinement
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx} -- after refinement")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)
```
Propogate the prompts again to get masklet across the video
```
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
```
**Segment an object using box prompt**

If you have previously run interefence_state, you must reset it using the code:
```
predictor.reset_state(inference_state)
```
This is creating the box prompt on frame 0
```
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([300, 0, 500, 400], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca()
```
Further refine the segmentation mask from a box prompt
```
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (460, 60) to refine the mask
points = np.array([[460, 60]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
# note that we also need to send the original box input along with
# the new refinement click together into `add_new_points_or_box`
box = np.array([300, 0, 500, 400], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    box=box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
```
Then get the masklet throughout the entire video and propogate the prompts
```
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
```
** Segment multiple objects **

If you have previously run interefence_state, you must reset it using the code:
```
predictor.reset_state(inference_state)
```
Add two objects on a frame
```
prompts = {}  # hold all the clicks we add for visualization
```
Select the Frame and assign an object id
```
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (200, 300) to get started on the first object
points = np.array([[200, 300]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
```
Negative click to refine the mask, for this situation, only have the mask over the boys shirt
```
# add the first object
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[200, 300], [275, 175]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
```
Create an object id and select your frame to add a positive click for another object, for this situation, the shirt of the kid on the right
```
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[400, 150]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels

# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
```
Propogate the prompts to get maskets across the video

```
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
```


## Troubleshooting
**Module conda does not exist**
- If your conda functions "do not exist" due to you having existing anaconda setup files stored in another directory, download miniconda
  ```
  curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
  start /wait "" .\miniconda.exe /S
  del .\miniconda.exe
  ```
**Jupyter Notebook not found**
- Simply run the command
  ```
  conda install jupyter
  jupyter-notebook
  ```
**Cannot successfully run the checkpoint bash script**
- Open the file manager and download the checkpoints list https://github.com/facebookresearch/sam2?tab=readme-ov-file and save in checkpoints folder

**Module torch/numpy/matplotlib not found**
- This error arises when the setup file within video_predictor_example fails to run. Modify the cell in the jupyter notebook to say:
```
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
!conda install numpy -y
import numpy as np
!conda install pytorch -y
import torch
!conda install matplotlib -y
import matplotlib.pyplot as plt
!conda install Pillow -y
from PIL import Image

```

