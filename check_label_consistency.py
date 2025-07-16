#!/usr/bin/env python3
"""
Check and fix label consistency between pretrained model and new dataset
"""

import json
import argparse
import os


def analyze_pretrained_model_classes():
    """Show the classes used in your pretrained model"""
    print("🔍 Pretrained Model Classes (from your original code):")
    print("=" * 50)
    
    pretrained_classes = {
        0: "background", 
        1: "kaiko_live", 
        2: "kaiko_pao1", 
        3: "kaiko_rhi"
    }
    
    for class_id, class_name in pretrained_classes.items():
        print(f"  {class_id}: {class_name}")
    
    print(f"\nTotal classes: {len(pretrained_classes)} (including background)")
    return pretrained_classes


def analyze_new_dataset(annotation_file):
    """Analyze the classes in your new dataset"""
    print("\n🔍 New Dataset Classes:")
    print("=" * 50)
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    categories = data['categories']
    print(f"Categories found: {len(categories)}")
    
    for cat in categories:
        print(f"  {cat['id']}: {cat['name']}")
    
    # Check what category_ids are actually used in annotations
    used_category_ids = set()
    for ann in data['annotations']:
        used_category_ids.add(ann['category_id'])
    
    print(f"\nCategory IDs used in annotations: {sorted(used_category_ids)}")
    return categories, used_category_ids


def create_updated_annotation_file(input_file, output_file, mapping_strategy="extend"):
    """
    Create updated annotation file with consistent labels
    
    mapping_strategy options:
    - "extend": Add new 'kaiko' as class 4, keep existing classes 1,2,3 for future use
    - "replace": Map 'kaiko' to 'kaiko_live' (class 1)
    - "multi_class": Create separate categories for different silkworm types
    """
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n🔄 Applying mapping strategy: {mapping_strategy}")
    
    if mapping_strategy == "extend":
        # Keep your new 'kaiko' as a separate class (class 4)
        # This allows future expansion to distinguish kaiko_live, kaiko_pao1, kaiko_rhi
        new_categories = [
            {"id": 1, "name": "kaiko_live", "supercategory": "silkworm"},
            {"id": 2, "name": "kaiko_pao1", "supercategory": "silkworm"}, 
            {"id": 3, "name": "kaiko_rhi", "supercategory": "silkworm"},
            {"id": 4, "name": "kaiko", "supercategory": "silkworm"}  # Your new annotations
        ]
        
        # Update annotations: map category_id 1 -> 4
        for ann in data['annotations']:
            if ann['category_id'] == 1:  # Original 'kaiko'
                ann['category_id'] = 4   # New 'kaiko' class
        
        print("  ✅ Mapped 'kaiko' to class 4")
        print("  ✅ Reserved classes 1,2,3 for kaiko_live, kaiko_pao1, kaiko_rhi")
        
    elif mapping_strategy == "replace":
        # Map your 'kaiko' to 'kaiko_live' (class 1)
        new_categories = [
            {"id": 1, "name": "kaiko_live", "supercategory": "silkworm"},
            {"id": 2, "name": "kaiko_pao1", "supercategory": "silkworm"},
            {"id": 3, "name": "kaiko_rhi", "supercategory": "silkworm"}
        ]
        
        # Annotations stay the same (already class 1)
        print("  ✅ Mapped 'kaiko' to 'kaiko_live' (class 1)")
        
    elif mapping_strategy == "multi_class":
        # Create a comprehensive set for future annotation
        new_categories = [
            {"id": 1, "name": "kaiko_live", "supercategory": "silkworm"},
            {"id": 2, "name": "kaiko_pao1", "supercategory": "silkworm"},
            {"id": 3, "name": "kaiko_rhi", "supercategory": "silkworm"},
            {"id": 4, "name": "kaiko_general", "supercategory": "silkworm"}  # For unlabeled stage
        ]
        
        # Map current annotations to general class
        for ann in data['annotations']:
            if ann['category_id'] == 1:
                ann['category_id'] = 4  # kaiko_general
        
        print("  ✅ Mapped 'kaiko' to 'kaiko_general' (class 4)")
        print("  ✅ Set up classes for future stage-specific annotation")
    
    # Update the dataset
    data['categories'] = new_categories
    
    # Save updated file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  💾 Updated annotation file saved: {output_file}")
    
    return new_categories


def create_training_config(num_classes, strategy):
    """Create a training configuration based on the chosen strategy"""
    
    config_template = f"""
# Training Configuration for Silkworm Detection
# Strategy: {strategy}
# Number of classes: {num_classes} (including background)

TRAINING_CONFIG = {{
    'num_classes': {num_classes},  # Including background class
    'strategy': '{strategy}',
    'class_mapping': {{
"""
    
    if strategy == "extend":
        config_template += """        0: 'background',
        1: 'kaiko_live',
        2: 'kaiko_pao1', 
        3: 'kaiko_rhi',
        4: 'kaiko'  # Your new annotations
    }
}

# For inference, you can map detections back:
# - Class 4 detections are your newly annotated silkworms
# - Classes 1,2,3 are from pretrained model (if any remain)
"""
    elif strategy == "replace":
        config_template += """        0: 'background',
        1: 'kaiko_live',  # Your annotations mapped here
        2: 'kaiko_pao1',
        3: 'kaiko_rhi'
    }
}

# All your annotations are treated as 'kaiko_live'
"""
    elif strategy == "multi_class":
        config_template += """        0: 'background',
        1: 'kaiko_live',
        2: 'kaiko_pao1',
        3: 'kaiko_rhi', 
        4: 'kaiko_general'  # Your current annotations
    }
}

# Future: Re-annotate class 4 into classes 1,2,3 as needed
"""
    
    return config_template


def main():
    parser = argparse.ArgumentParser(description='Check and Fix Label Consistency')
    parser.add_argument('--annotation-file', 
                       default='kaiko_labeled/annotations/instances_default.json',
                       help='Your new annotation file')
    parser.add_argument('--output-file',
                       default='work_dirs/detection_data/instances_consistent.json', 
                       help='Output file with consistent labels')
    parser.add_argument('--strategy', 
                       choices=['extend', 'replace', 'multi_class'],
                       default='extend',
                       help='Label mapping strategy')
    
    args = parser.parse_args()
    
    print("🔍 Checking Label Consistency")
    print("=" * 60)
    
    # Analyze pretrained model classes
    pretrained_classes = analyze_pretrained_model_classes()
    
    # Analyze new dataset classes  
    new_categories, used_ids = analyze_new_dataset(args.annotation_file)
    
    print(f"\n⚠️ Inconsistency Detected:")
    print(f"  Pretrained model expects: {list(pretrained_classes.values())}")
    print(f"  New dataset has: {[cat['name'] for cat in new_categories]}")
    
    print(f"\n💡 Recommended Strategy: '{args.strategy}'")
    
    if args.strategy == "extend":
        print("  - Keeps all existing classes from pretrained model")
        print("  - Adds your 'kaiko' as a new class (class 4)")
        print("  - Best for: Gradual expansion of dataset")
        num_classes = 5  # 0,1,2,3,4
        
    elif args.strategy == "replace":
        print("  - Maps your 'kaiko' to 'kaiko_live' (class 1)")
        print("  - Assumes all your silkworms are 'live' stage")
        print("  - Best for: Simple binary detection (background vs silkworm)")
        num_classes = 4  # 0,1,2,3
        
    elif args.strategy == "multi_class":
        print("  - Sets up comprehensive class structure")
        print("  - Your annotations become 'kaiko_general'")
        print("  - Best for: Future detailed stage annotation")
        num_classes = 5  # 0,1,2,3,4
    
    # Create updated annotation file
    updated_categories = create_updated_annotation_file(
        args.annotation_file, 
        args.output_file, 
        args.strategy
    )
    
    # Show training configuration
    print(f"\n📋 Training Configuration:")
    config = create_training_config(num_classes, args.strategy)
    print(config)
    
    # Save config to file
    config_file = args.output_file.replace('.json', '_config.py')
    with open(config_file, 'w') as f:
        f.write(config)
    print(f"💾 Training config saved: {config_file}")
    
    print(f"\n✅ Label consistency check completed!")
    print(f"📁 Use this file for training: {args.output_file}")
    print(f"🎯 Number of classes for model: {num_classes}")


if __name__ == '__main__':
    main()
