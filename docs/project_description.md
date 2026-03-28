# AquaDet: Project Description

## 1. The Problem
Millions of tons of synthetic macro-plastics and solid organic waste enter global aquatic ecosystems annually, posing a catastrophic threat to marine biology and eventually entering the human food chain. To combat this, environmental agencies employ Autonomous Underwater Vehicles (AUVs) to patrol and collect debris. However, the operational efficiency of these robotic systems is severely bottlenecked by their "eyes"—their perception AI. 

Standard terrestrial AI models (like YOLO or Computer Vision algorithms) are designed for operation in clear air. When deployed underwater, they dramatically fail due to complex optical physics: **light attenuation** (water absorbs the red/yellow light spectrum, turning everything blue) and **backscatter** (light reflecting off suspended dirt and micro-organisms, causing severe visual "murkiness" or haze). As a result, standard autonomous drones cannot accurately detect, classify, or estimate the size of the waste they need to collect, either missing the trash entirely or lacking the geometric data to grasp it safely.

## 2. The Goal
The primary goal of the *AquaDet* project is to engineer a highly robust, computationally lightweight **Visual Intelligence Cortex** for underwater drones. 

Specifically, the project aims to:
1. Dynamically clean and restore the real colors of underwater video feeds in real-time, removing the "murkiness" caused by water physics.
2. Accurately detect, categorize (Plastic, Metal, Organic, Microplastic), and precisely track moving trash.
3. Mathematically estimate the absolute real-world physical size of the detected objects (in millimeters) using only a single standard camera lens, eliminating the need for expensive, fragile stereoscopic 3D cameras or LiDAR systems.
4. Ensure the entire algorithmic pipeline operates fast enough (30+ FPS) to fit aboard heavily resource-constrained micro-computers attached to AUVs (like the NVIDIA Jetson series).

## 3. Concept of the Solution
*AquaDet* solves these challenges by combining advanced Deep Learning with classic Optical Physics in a unified, multi-task neural network architecture. 

The concept revolves around the following operational pipeline:
*   **Physics-Grounded Restoration (PI-GE):** Before the AI even attempts to look for trash, the incoming camera frame passes through a Physical-Image-Geometry Enhancement (PI-GE) module. This mathematical gate dynamically predicts how much light is being lost to the water's depth and dirt density, effectively "inverting" the water. The AI removes the haze on the fly, feeding a clear image to the rest of the neural network.
*   **Multi-Task Processing (4D Mapping):** Instead of running three separate heavy algorithms (one for finding boxes, one for making masks, one for guessing distance), the *AquaDet* architecture uses a shared "Brain" (Hybrid Convolutional Backbone + BiFPN) that splits into parallel lightweight tasks. It maps the 2D bounding box, the exact shape (segmentation mask), and the specific object class simultaneously.
*   **Monocular Size Geometry (ASPP):** To figure out how big an object is using only one camera, the network utilizes Atrous Spatial Pyramid Pooling (looking at the image through artificially widened "lenses" of different scales). By understanding visual context (e.g., how the object sits against the distant ocean floor), it calculates a highly accurate depth projection ($Z$). Using the camera's hardware focal length ($f$), the AI computes the final physical geometry of the debris natively: $Size = Size_{pixel} \times \frac{f}{Z}$.
*   **Edge-Optimization:** The entire concept is mathematically quantized (compressed) and mapped for NVIDIA TensorRT compilation, guaranteeing that these complex mathematical calculations do not drain the underwater drone's battery or delay physical robotic arm collection mechanics.
