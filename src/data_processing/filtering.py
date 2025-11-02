import json
import shutil
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
from src.utils import extract_frames

def filter_json_files(source_dir: str, output_dir: str):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_path.resolve()}: {e}")
        return
    
    files_processed = 0
    files_copied = 0

    for file_path in tqdm(source_path.glob('*.json')):
        files_processed += 1
        try:
            with file_path.open('r', encoding='utf-8') as f:
                scene = json.load(f)
            frames = extract_frames(scene)

            heading = get_nested_value(frames, [0, 'ego', 'features', 'heading'])

            if heading is not None:
                files_copied += 1
                destination_file = output_path / file_path.name
                shutil.copy2(file_path, destination_file)
            else:
                pass

        except json.JSONDecodeError:
            print(f"  [ERROR] Skipping {file_path.name}: Invalid JSON format.")
        except (KeyError, IndexError, TypeError):
            print(f"  [ERROR] Skipping {file_path.name}: Expected structure (e.g., 'frames[0]') not found.")
        except Exception as e:
            print(f"  [ERROR] Skipping {file_path.name}: An unexpected error occurred: {e}")

    print(f"\n--- Filter Complete ---")
    print(f"Total files scanned: {files_processed}")
    print(f"Files copied:      {files_copied}")
    print(f"Files skipped:     {files_processed - files_copied}")

def get_nested_value(data: dict, keys: list) -> Optional[Any]:
    current_level = data
    for key in keys:
        if isinstance(current_level, dict):
            current_level = current_level.get(key)
        elif isinstance(current_level, list):
            try:
                current_level = current_level[key]
            except (IndexError, TypeError):
                return None
        else:
            return None
        
        if current_level is None:
            return None
            
    return current_level
