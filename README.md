# FlowMatch Euler Discrete Scheduler for ComfyUI

**FlowMatchEulerDiscrete** seems not exposed in comfy, but it is what the official demo in diffusers use.
So I am exposing it in the scheduler section and provide a node, experimental, to configure the scheduler for use with CustomSampler
So...if you want **sharper and more noise free images**, you can experiment with this.

## Installation
this is an alpha stage node, so not esposed yet in manager. to install
open a terminal command in your custom nodes then, you do have git installed I assume, then...:

```git clone https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler.git```



<img width="1792" height="1120" alt="example" src="https://github.com/user-attachments/assets/91d4f679-9c9e-40ef-bed6-12bb860c37e7" />


Custom node that exposes all parameters of the FlowMatchEulerDiscreteScheduler. Outputs SIGMAS for use with **SamplerCustom** node.

![highlight](https://github.com/user-attachments/assets/249cd15d-f373-46c7-bacb-13a4b5421ba3)

## Usage

1. Add **FlowMatch Euler Discrete Scheduler (Custom)** node to your workflow
2. Connect its SIGMAS output to **SamplerCustom** node's sigmas input
3. Adjust parameters to control the sampling behavior

## Example Workflow

```
Model -> SamplerCustom
         ^ 
         |
         FlowMatch Euler Scheduler (steps=30, use_karras_sigmas=True)
```

## Tech bits:
- https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete
- https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json
