## Time and Requirement Analysis

This document explains the time and requirement analysis of the unified model.

📝 An RTX 3090 GPU was used for the time calculations

### Time Analysis

Since we are using couple of foundation model to anonymize images, the tool might take significant amount of time.

You can see which model takes how much time per image in the following table:

| Model Name         | Time per Image (ms)          |
| ------------------ | ---------------------------- |
| Grounding DINO     | `~300 ms`                    |
| Segment Anything 2 | `~130 ms`                    |
| Open CLIP          | `~30 ms` x `number of boxes` |
| YOLO v11 X         | `~20 ms `                    |

**Note:** You can blur the bounding boxes instead of masks to reduce the time. When you blur the bounding boxes, `Segment Anything 2` model will not be used. In configuration file, you can set `blur.region` to `bbox` to blur the bounding boxes.

**Note:** You can close `OpenClip` model to reduce the time. When you close the `OpenClip` model, in validation step, the model will not be used. In configuration file, you can set `open_clip.run` to `False` to close the model.

### Requirement Analysis

You can see how much VRAM is required for each model in the following table:

| Model Name         | VRAM Requirement (MiB) |
| ------------------ | ---------------------- |
| Grounding DINO     | `~1200 MiB`            |
| Segment Anything 2 | `~448 MiB`             |
| Open CLIP          | `~930 MiB`             |
| YOLO v11 X         | `~864 MiB`             |