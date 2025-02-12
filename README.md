# Homework-2
## Key Benefits and Features
**Components**
- Image Encoder
    - It is a hierarchical masked autoencoder which allows the use of multiscale features during decoding. Thus, it encodes the video frames one frame at a time
- Memory Attention
    - A cross-attention mechanism conditioning the current frames' features using previous frames, predictions and prompts.
- Prompt Encoder
    - Handles different types of prompts such as sparse prompts(represented by positional encodings and learned embeddings) and dense prompts(handled by convolutional layers).
- Mask Encoder
    - The mask decoder uses the encoded prompts, and the encoded and conditioned frames from the memory attention module to generate the mask for the current frame. This helps create high-resolution information for decoding
- Memory Encoder
    - This uses the encodes the current frame’s predictions and the embeddings from the image encoder to be used in the future.
- Memory Bank
    - It stores the information about past predictions for the target object in the video. Additionally, it maintains the history of prompts from the prompt encoder for the prompted frames
**Data Engine**
Create data engine with model in the loop
- Phase 1
   - First, SAM is used to generate masks and assist humans with annotations. This is then followed by human annotators that correct and refine the annotations on frames of videos  with pixel-precise editing tools.
- Phase 2
    - The mask from SAM 1 and manual masks are used as prompts for the SAM 2 model. Using the prompted masks from the first frame, subsequent masks are created by the SAM 2 model
- Phase 3
    - Only SAM 2 is used and accepts all types of prompts
**Benefits**
- Minimal human intervention is needed
- Precise auto annotations
- Good for object tracking and segmenting
## How to Install Code and Libraries
First, Open a New Terminal and Clone the Repository
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
Then, Install the Depednecies for the demo
```
pip install -e ".[demo]"

```
Then, Download all the Checkpoints

```
cd checkpoints
./download_ckpts.sh
```
Now create a new directory called "custom_code" to store all of the custom notebooks and scripts

## How to Run the Code



