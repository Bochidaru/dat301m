import json
import pandas as pd
import os
import sys
from typing import Dict, List, Any


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def convert_json_to_csv(json_file_path: str, output_csv_path: str) -> bool:
    """
    Convert JSON caption file to CSV format.
    
    Args:
        json_file_path: Path to input JSON file
        output_csv_path: Path to output CSV file
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    print(f"Processing {json_file_path}...")
    
    # Load JSON data
    data = load_json_data(json_file_path)
    if not data:
        return False
    
    # Extract images data
    images_data = []
    if 'images' in data:
        for img in data['images']:
            images_data.append({
                'id': img['id'],
                'file_name': img['file_name'],
                'flickr_url': img.get('flickr_url', ''),
                'coco_url': img.get('coco_url', ''),
                'height': img['height'],
                'width': img['width']
            })
    
    # Extract annotations data
    annotations_data = []
    if 'annotations' in data:
        for ann in data['annotations']:
            annotations_data.append({
                'image_id': ann['image_id'],
                'id_anno': ann['id'],
                'caption': ann['caption']
            })
    
    # Convert to DataFrames
    images_df = pd.DataFrame(images_data)
    annotations_df = pd.DataFrame(annotations_data)
    
    # Merge on image_id (from annotations) matching id (from images)
    merged_df = pd.merge(annotations_df, images_df, 
                        left_on='image_id', 
                        right_on='id', 
                        how='inner')
    
    # Select and reorder columns as requested
    final_columns = [
        'id',
        'file_name', 
        'flickr_url', 
        'coco_url', 
        'height', 
        'width',
        'image_id', 
        'id_anno', 
        'caption'
    ]
    
    result_df = merged_df[final_columns]
    
    # Save to CSV
    try:
        result_df.to_csv(output_csv_path, index=False)
        print(f"Successfully converted to {output_csv_path}")
        print(f"Total records: {len(result_df)}")
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False


def main():
    """Main function to process all JSON files in the current directory."""
    # current_dir = os.getcwd()
    # json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]
    json_files = ["captions_val2017_trans.json", "captions_train2017_trans.json"]
    
    if not json_files:
        print("No JSON files found in current directory")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        print(f"  - {file}")
    
    # Process each JSON file
    successful_conversions = 0
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        csv_file = f"{base_name}_converted.csv"
        
        if convert_json_to_csv(json_file, csv_file):
            successful_conversions += 1
    
    print(f"\nConversion complete! {successful_conversions}/{len(json_files)} files converted successfully.")


if __name__ == "__main__":
    main()
