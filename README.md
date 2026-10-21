# Adversarial Pattern Lab

Generate patterns, apply them to a person's clothing through a mask, and measure how a **local** Edge Impulse (`.eim`) detector responds. Everything runs on your machine.

<img width="2158" height="1392" alt="Diagrama" src="https://github.com/user-attachments/assets/7faefc26-5f9c-4037-a4be-001513062bb9" />


*Developed by Roni Bandini · October 2026 · Maker (Counter)Culture*

---

## What it does

1. **Generates patterns.** 36 procedural, seeded patterns: textures, object shapes, camouflage and more.
2. **Applies them to a photo.** You supply a person image and a transparent PNG mask of the garments. The pattern is composited only inside the mask, keeping the photo's shading.
3. **Evaluates on a local model.** Each variant goes through your `.eim` model. The tool records the confidence for a target label (e.g. `person`) and whether it falls below your threshold.
4. **Shows the result.** The person with the pattern and the bounding box, the pattern itself, a results table, a log and a per-pattern detection rate.
5. **Exports the successful patterns.** A zip with the composites, the tested patches, 1200 px patches and a CSV.

## Interfaces

<img width="1024" height="485" alt="Interface" src="https://github.com/user-attachments/assets/cf084ff5-92b4-478d-b6f4-a97b093be4d5" />


## Requirements

- Python 3.9 or newer.
- **Linux, WSL2 or a Linux container** for the `.eim` model (an Edge Impulse limitation). The pattern preview works elsewhere, but running an evaluation needs Linux.
- An Edge Impulse **object detection or classification** model exported as a Linux `.eim` built for your CPU architecture.
- A person photo and a **garment mask**: a transparent PNG where the garments are opaque, the **same size** as the photo.

## Install

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers all three interfaces. For the vintage terminal UI alone you only need `opencv-python`, `numpy`, `pandas` and `edge_impulse_linux`.

## Run

```bash
python app.py \
  --model models/person.eim \
  --image images/person4.png \
  --mask  images/mask.png
```

Then open <http://127.0.0.1:8765> (it opens automatically; on WSL2 open the URL in your Windows browser). The server listens on `127.0.0.1` only.

| Option | Default | Meaning |
|---|---|---|
| `--model` | `models/person.eim` | Edge Impulse `.eim` model |
| `--image` | `images/person4.png` | Person photo |
| `--mask` | `images/mask.png` | Transparent PNG garment mask |
| `--label` | `person` | Label whose confidence is measured (empty = any) |
| `--threshold` | `0.65` | Confidence below this counts as "not detected" |
| `--variants` | `3` | Seeds per pattern |
| `--size` | `300` | Pattern tile size in pixels |
| `--seed` | `0` | Base random seed |
| `--place` | `repeat` | `repeat`, `mirror` or `stretch` |
| `--port` | `8765` | Local port |

### Keys (vintage terminal UI)

| Key | Action |
|---|---|
| `F5` | Run the evaluation |
| `F6` | Export the successful patterns (zip) |
| `F7` | Green-monochrome pictures |
| `↑` / `↓` | Browse results |

Click a pattern name to toggle it, a category header to toggle the whole category, and a table row to inspect that run.

## How the evaluation works

- `EimDetector.detect()` (`eim_detector.py`) is the only place that talks to Edge Impulse. It returns detections as plain dicts, from either `bounding_boxes` (object detection) or `classification` output.
- `selectTarget()` keeps detections matching your label and picks the largest box (or the highest confidence).
- A run counts as **not detected** when that confidence is below the threshold. If the model never reports your label, the confidence is 0, so check the label printed at start-up (`labels=[...]`) against `--label`.
- Boxes are mapped back from the model's input space onto your original image, handling Edge Impulse's squash, fit-shortest and fit-longest resize modes. If the model info does not report a resize mode, squash is assumed.

## Pattern library

36 patterns in seven groups. Hover a pattern in the UI for its rationale and reference.

| Group | Patterns |
|---|---|
| Baseline geometric | blocks, stripes, checker, rings, smooth, voronoi, waves, noise, polka, hexagons, chevron |
| Natural textures | leopard, zebra, giraffe, cow, fish scales, fur, flowers |
| Object shapes | eyes, faces, octagons, traffic lights, wheels, clocks, vehicles, glyphs, barcode, QR-like |
| Camouflage | dazzle, digital, woodland |
| Optimisation-inspired | spiral, saturated blobs, multi-channel gratings, bullseye |
| Decoy | stick figures (person-like cues that may *raise* detection) |

All shapes are generic and drawn procedurally. There are no logos, brands or copyrighted characters. You can also add your own textures.

## Exported files

`successful_patterns.zip` contains only runs below the threshold:

```
composites/<pattern>_s<seed>.png     person + pattern, as tested
patches/<pattern>_s<seed>_tested.png the exact patch that was tested
patches/<pattern>_s<seed>_1200.png   same pattern and seed regenerated at 1200 px
successful.csv                       pattern, category, seed, conf, delta
```

Some patterns depend on absolute pixel frequencies, so the 1200 px version can differ from the tested one. The `_tested` patch is always exact.

## Limitations

- Needs a **precise mask**. The tool does not segment garments; patterns land exactly where the mask is opaque.
- Patterns are flat 2D overlays on a single photo. They are not a physical, printed or multi-view test, so results do not transfer to the real world.

## Background reading

- Brown et al., *Adversarial Patch*, 2017
- Thys et al., *Fooling automated surveillance cameras: adversarial patches to attack person detection*, CVPRW 2019
- Xu et al., *Adversarial T-shirts to Evade ML Person Detectors*, ECCV 2020
- Hu et al., *Naturalistic Physical Adversarial Patch for Object Detectors*, ICCV 2021
- Hu et al., *Adversarial Texture for Fooling Person Detectors in the Physical World*, CVPR 2022
- Geirhos et al., *ImageNet-trained CNNs are biased towards texture*, ICLR 2019

