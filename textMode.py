# Adversarial Pattern Lab
# Roni Bandini, Buenos Aires - Argentina - 10/2026 
# GPL License
# Text mode version 

import argparse
import os
import sys
import cv2
import numpy as np
from eim_detector import EimDetector
from eval_core import load_mask, run_evaluation
from pattern_library import REGISTRY

def clearScreen():
    os.system("cls" if os.name == "nt" else "clear")

def printBanner():
    print("=" * 78)
    print(" ADVERSARIAL PATTERN LAB                                      [TEXT MODE]")
    print(" Computer Vision Evaluation Toolkit")
    print("=" * 78)
    print(" Author : Roni Bandini")
    print(" Location: Buenos Aires, Argentina")
    print(" Release : October 2026  |  License: GPL")
    print("-" * 78)


def main():
    clearScreen()
    printBanner()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/person.eim")
    parser.add_argument("--image", default="images/Brad.png")
    parser.add_argument("--mask", default="images/BradMask.png")
    parser.add_argument("--label", default="person")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--place", default="repeat")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--exportDir", default="output_results")
    args = parser.parse_args()

    modelPath = args.model
    imagePath = args.image
    maskPath = args.mask
    label = args.label
    threshold = args.threshold
    variants = args.variants
    size = args.size
    seed = args.seed
    place = args.place
    selectedPatterns = args.patterns
    exportDir = args.exportDir

    if exportDir and not os.path.exists(exportDir):
        os.makedirs(exportDir)

    if not selectedPatterns:
        selectedPatterns = [name for name, spec in REGISTRY.items() if spec.category != "baseline_geometric"]

    personImg = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if personImg is None:
        print(f"Error: cannot read image {imagePath}")
        sys.exit(1)
    if personImg.ndim == 3 and personImg.shape[2] == 4:
        personImg = cv2.cvtColor(personImg, cv2.COLOR_BGRA2BGR)

    maskImg = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    if maskImg is None:
        print(f"Error: cannot read mask {maskPath}")
        sys.exit(1)
    
    maskFloat = load_mask(maskImg, personImg.shape)
    pixelVal = maskImg[0, 0]
    if isinstance(pixelVal, np.ndarray):
        pixelVal = pixelVal[0]
    if pixelVal > 127:
        maskFloat = 1.0 - maskFloat

    detector = EimDetector(modelPath)
    totalRuns = len(selectedPatterns) * variants
    print(f"\nStarting evaluation  |  Patterns: {len(selectedPatterns)}  |  Variants each: {variants}  |  Total runs: {totalRuns}")
    print(f"Model: {modelPath}  |  Label: {label}  |  Detection threshold: {threshold:.3f} ({threshold * 100:.1f}%)")
    print("─" * 86)
    print(f"{'PATTERN':<22} {'SEED':>6} {'CONFIDENCE':>12} {'DELTA':>11} {'NOT DETECTED':>14}")
    print("─" * 86)

    def handleRecord(record):
        notDetectedMark = "*" if not record["detected"] else ""
        if record["kind"] == "baseline":
            print(f"{'baseline':<22} {'-':>6} {record['conf'] * 100:>10.1f}% {'-':>11} {notDetectedMark:>14}")
            fileName = "baseline_annotated.png"
        else:
            print(f"{record['pattern']:<22} {record['seed']:>6} {record['conf'] * 100:>10.1f}% {record['delta'] * 100:>+9.1f}pp {notDetectedMark:>14}")
            fileName = f"{record['pattern']}_s{record['seed']}_annotated.png"

        if exportDir:
            outPath = os.path.join(exportDir, fileName)
            cv2.imwrite(outPath, record["annotated"])

    try:
        run_evaluation(
            detector=detector,
            person=personImg,
            mask=maskFloat,
            pattern_names=selectedPatterns,
            variants=variants,
            size=size,
            seed=seed,
            tile_mode=place,
            label=label,
            threshold=threshold,
            resize_mode=detector.info.get("model_parameters", {}).get("image_resize_mode"),
            on_record=handleRecord
        )
        print("─" * 86)
        print(f"Evaluation complete. Annotated images saved to {exportDir}/")
    finally:
        detector.close()

if __name__ == "__main__":
    main()