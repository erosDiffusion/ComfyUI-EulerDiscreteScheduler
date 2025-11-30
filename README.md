# FlowMatch Euler Discrete Scheduler for ComfyUI

9 steps, big res, zero noise.

**FlowMatchEulerDiscrete** seems not exposed in ComfyUI, but it is what the official Z-Image demo in diffusers use.

So:

- I am exposing it in the scheduler section for you to use within KSampler.
- On top I provide a node, experimental, to configure the scheduler for use with CustomSampler and play with.

In short...if you want **sharper and noise free images**, use this!

## Installation

- use comfy ui manager (search erosDiffusion or ComfyUI-EulerFlowMatchingDiscreteScheduler)

or

- `git clone https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler.git` in your custom nodes folder.

Example output (more below)
<img width="1792" height="1120" alt="example" src="https://github.com/user-attachments/assets/91d4f679-9c9e-40ef-bed6-12bb860c37e7" />

## What you get

- one new scheduler **FlowMatchEulerDiscreteScheduler** registered in the KSampler
- a custom node that exposes all parameters of the FlowMatchEulerDiscreteScheduler which Outputs **SIGMAS** for use with **SamplerCustom** node.

![highlight](https://github.com/user-attachments/assets/249cd15d-f373-46c7-bacb-13a4b5421ba3)

## Usage

- **Simple**: select the FlowMatchEulerDiscreteScheduler in the default workflow from ComfyUI and run.
- **Advanced/experimental**:
  1. Add **FlowMatch Euler Discrete Scheduler (Custom)** node to your workflow
  2. Connect its SIGMAS output to **SamplerCustom** node's sigmas input
  3. Adjust parameters to control the sampling behavior

## Tech bits:

- https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete
- https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json

## Find this useful and want to support ?

[Buy me a beer!](https://donate.stripe.com/cNi9ALaASf65clXahPcV201)

<img width="1920" height="1088" alt="ComfyUI_00716_" src="https://github.com/user-attachments/assets/bb2fc530-8a90-4180-96fb-adf6c48f5b09" />

More examples:
<img width="1536" height="1088" alt="image" src="https://github.com/user-attachments/assets/1ab561e7-b51d-413c-b788-13ed4fb6e129" />
<img width="1536" height="1088" alt="image" src="https://github.com/user-attachments/assets/1931af7e-1b3e-47c9-ac20-27add5135a71" />

## Changelog
**1.0.4**
add start and end step by etupa, with some fixes
**1.0.3**

- node publish action

**1.0.2**

- changed the device management in the custom scheduler node to be on gpu (cuda)
- removed flash attention node dependency from the custom scheduler node
- removed flash attention node from init
- added mit licensing
