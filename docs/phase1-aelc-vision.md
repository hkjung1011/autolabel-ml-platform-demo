# Phase 1: Auto Exposure + Lighting Control + Vision

## Goal

Make image capture more usable for detection and segmentation before relying on recovery transforms.

## Core loop

1. Receive RGB, RAW, or multi-exposure input
2. Measure low-light, backlight, blur, noise, clipping, and shadow crushing
3. Produce exposure and lighting recommendations
4. Estimate `vision_ready_score`
5. Run enhancement only when capture control is not enough or when processing offline data
6. Pass controlled or enhanced variants into auto-labeling and review

## Inputs

- Image pixels
- Optional camera metadata
  - ISO / analog gain
  - shutter time
  - exposure bias
  - white balance
  - lens info
- Optional lighting rig metadata
  - brightness channels
  - color temperature
  - lighting zone id

## Outputs

- `target_exposure_bias`
- `recommended_gain`
- `recommended_shutter_ratio`
- `lighting_action`
- `vision_ready_score`
- explanation string for operators and audit logs

## Why this should be first

- Better capture conditions reduce downstream enhancement artifacts
- Label confidence improves when exposure is stabilized upstream
- Lighting control is a product differentiator that common labeling tools do not own
- The same control loop can support cameras, inspection rigs, and smart factories later

## Practical Phase 1 scope

- Offline image analysis with recommendation output
- Rule-based exposure and lighting policy
- Dashboard for operator review
- Audit log of recommendation and selected pipeline

## Phase 1.5

- Camera SDK integration
- Lighting controller integration
- Real-time feedback loop during acquisition
- Per-device calibration profiles
