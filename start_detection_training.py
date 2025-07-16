#!/usr/bin/env python3
"""
Simple script to start detection training with proper label consistency
"""

import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Start Detection Training')
    parser.add_argument('--strategy', choices=['extend', 'replace'], default='extend',
                       help='Label mapping strategy')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Detection Training")
    print(f"Strategy: {args.strategy}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Set number of classes based on strategy
    if args.strategy == 'extend':
        num_classes = 5  # background + kaiko_live + kaiko_pao1 + kaiko_rhi + kaiko
        print("Classes: background, kaiko_live, kaiko_pao1, kaiko_rhi, kaiko (your new data)")
    else:  # replace
        num_classes = 4  # background + kaiko_live + kaiko_pao1 + kaiko_rhi
        print("Classes: background, kaiko_live (your data), kaiko_pao1, kaiko_rhi")
    
    # Prepare consistent labels if not already done
    print("\n📋 Preparing consistent labels...")
    prep_cmd = [
        'python', 'check_label_consistency.py',
        '--strategy', args.strategy
    ]
    subprocess.run(prep_cmd)
    
    # Start training
    print(f"\n🎯 Starting training with {num_classes} classes...")
    train_cmd = [
        'python', 'train_detection.py',
        '--annotation-file', 'work_dirs/detection_data/instances_consistent.json',
        '--num-classes', str(num_classes),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--output-dir', f'work_dirs/detection_training_{args.strategy}'
    ]
    
    print("Command:", ' '.join(train_cmd))
    subprocess.run(train_cmd)


if __name__ == '__main__':
    main()
