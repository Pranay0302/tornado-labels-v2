"""Generate automatic segmentation proposals using SAMGeo."""

from __future__ import annotations

import argparse
import gc
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely.geometry as geom
from PIL import Image
from samgeo import SamGeo
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import ensure_directory, is_blank_tile

"""
Usage:
  python 03_samgeo_propose.py <tiles_dir> <out_geojson_dir> [--max-tiles N] [--timeout T] [--model MODEL]

Notes:
- Uses SAM default ViT (downloaded on first run).
- Produces polygons per tile as GeoJSON (pixel coords).
- Annotators can refine in X-AnyLabeling; you'll unify labels later.
- Optimized for performance with progress tracking and timeout handling.
"""

# Global timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timeout")

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def mask_to_polygons(mask):
    # Simple contour-to-polygon (non-georeferenced; pixel coords)
    # You may replace with skimage.measure.find_contours for better fidelity.
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) >= 3 and cv2.contourArea(c) > 128:  # ignore tiny blobs
            pts = c[:,0,:].astype(float).tolist()
            polys.append(geom.Polygon(pts).buffer(0))
    return polys

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 0

def process_tile_with_timeout(sam, img_np, img_fp, timeout_seconds=60):
    """Process a single tile with timeout and error handling"""
    try:
        with timeout(timeout_seconds):
            start_time = time.time()
            # Use mask_generator directly for automatic mask generation
            if sam.mask_generator is None:
                return None, "No mask generator available"
            
            masks = sam.mask_generator.generate(img_np)
            process_time = time.time() - start_time
            
            if len(masks) == 0:
                return None, f"No masks generated in {process_time:.1f}s"
            
            # Convert masks to the expected format (list of numpy arrays)
            mask_arrays = [mask['segmentation'] for mask in masks]
            
            return mask_arrays, f"Generated {len(mask_arrays)} masks in {process_time:.1f}s"
            
    except TimeoutError:
        return None, f"Timeout after {timeout_seconds}s"
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_sam_proposals(
    tiles_dir,
    out_dir,
    max_tiles=None,
    timeout_seconds=60,
    model_type="vit_h",
):
    """Generate automatic polygon proposals for *tiles_dir* and write them to *out_dir*."""
    tiles_dir = Path(tiles_dir)
    out_dir = ensure_directory(out_dir)
    
    # Initialize SAM with specified model
    print(f"[INIT] Loading SAM model: {model_type}")
    sam = SamGeo(model_type=model_type)
    
    # Get all image files
    image_files = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg")))
    total_files = len(image_files)
    
    if max_tiles:
        image_files = image_files[:max_tiles]
        print(f"[CONFIG] Processing first {max_tiles} tiles out of {total_files}")
    
    print(f"[CONFIG] Timeout per tile: {timeout_seconds}s")
    print(f"[CONFIG] GPU Memory before processing: {get_gpu_memory_usage()}MB")
    
    # Statistics
    stats = {
        'processed': 0,
        'skipped_blank': 0,
        'timeout': 0,
        'error': 0,
        'success': 0,
        'total_masks': 0,
        'start_time': time.time()
    }
    
    # Process tiles with detailed progress tracking
    pbar = tqdm(image_files, desc="Processing tiles", unit="tile")
    
    for img_fp in pbar:
        try:
            # Load and check image
            img = Image.open(img_fp).convert("RGB")
            img_np = np.array(img)
            
            # Skip blank tiles
            if is_blank_tile(img_np):
                stats['skipped_blank'] += 1
                pbar.set_postfix({
                    'Processed': stats['processed'],
                    'Skipped': stats['skipped_blank'],
                    'Success': stats['success'],
                    'GPU': f"{get_gpu_memory_usage()}MB"
                })
                continue
            
            stats['processed'] += 1
            
            # Process tile with timeout
            masks, status_msg = process_tile_with_timeout(sam, img_np, img_fp, timeout_seconds)
            
            if masks is None:
                if "Timeout" in status_msg:
                    stats['timeout'] += 1
                    print(f"\n[TIMEOUT] {img_fp.name}: {status_msg}")
                else:
                    stats['error'] += 1
                    print(f"\n[ERROR] {img_fp.name}: {status_msg}")
                
                pbar.set_postfix({
                    'Processed': stats['processed'],
                    'Skipped': stats['skipped_blank'],
                    'Success': stats['success'],
                    'Timeout': stats['timeout'],
                    'GPU': f"{get_gpu_memory_usage()}MB"
                })
                continue
            
            # Convert masks to polygons
            gdf_rows = []
            for i, m in enumerate(masks):
                polys = mask_to_polygons(m)
                for p in polys:
                    if p.is_valid and not p.is_empty:
                        gdf_rows.append({"tile": img_fp.name, "proposal_id": i, "geometry": p})
            
            stats['total_masks'] += len(masks)
            
            # Save results
            if gdf_rows:
                gdf = gpd.GeoDataFrame(gdf_rows, geometry="geometry", crs=None)
                gdf.to_file(out_dir / f"{img_fp.stem}.geojson", driver="GeoJSON")
                stats['success'] += 1
                print(f"\n[SUCCESS] {img_fp.name}: {status_msg}, saved {len(gdf_rows)} polygons")
            else:
                print(f"\n[WARN] {img_fp.name}: {status_msg}, but no valid polygons found")
            
            # Update progress bar
            pbar.set_postfix({
                'Processed': stats['processed'],
                'Skipped': stats['skipped_blank'],
                'Success': stats['success'],
                'Masks': stats['total_masks'],
                'GPU': f"{get_gpu_memory_usage()}MB"
            })
            
            # Memory cleanup
            del img, img_np, masks, gdf_rows
            gc.collect()
            
        except Exception as e:
            stats['error'] += 1
            print(f"\n[FATAL] {img_fp.name}: {str(e)}")
            pbar.set_postfix({
                'Processed': stats['processed'],
                'Skipped': stats['skipped_blank'],
                'Success': stats['success'],
                'Error': stats['error'],
                'GPU': f"{get_gpu_memory_usage()}MB"
            })
    
    # Final statistics
    total_time = time.time() - stats['start_time']
    print(f"\n[COMPLETE] Processing finished in {total_time/60:.1f} minutes")
    print(f"[STATS] Processed: {stats['processed']}, Skipped: {stats['skipped_blank']}, Success: {stats['success']}")
    print(f"[STATS] Timeouts: {stats['timeout']}, Errors: {stats['error']}, Total masks: {stats['total_masks']}")
    print(f"[STATS] Average time per tile: {total_time/stats['processed']:.1f}s")
    print(f"[STATS] Final GPU memory: {get_gpu_memory_usage()}MB")
    print(f"[OK] Proposals written to: {out_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAMGeo proposal generation with optimisations")
    parser.add_argument("tiles_dir", help="Directory containing image tiles")
    parser.add_argument("out_dir", help="Output directory for GeoJSON files")
    parser.add_argument("--max-tiles", type=int, help="Maximum number of tiles to process (for testing)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per tile in seconds (default: 60)")
    parser.add_argument(
        "--model",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type (default: vit_h)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    print("[CONFIG] Starting SAMGeo processing")
    print(f"[CONFIG] Tiles directory: {args.tiles_dir}")
    print(f"[CONFIG] Output directory: {args.out_dir}")
    print(f"[CONFIG] Model: {args.model}")
    print(f"[CONFIG] Timeout: {args.timeout}s")
    if args.max_tiles:
        print(f"[CONFIG] Max tiles: {args.max_tiles}")

    generate_sam_proposals(
        args.tiles_dir,
        args.out_dir,
        max_tiles=args.max_tiles,
        timeout_seconds=args.timeout,
        model_type=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
