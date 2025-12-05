#!/bin/bash
"""
Parallel Training Launcher
Trains navigation and fetch navigation simultaneously
Usage:
  python launch_parallel_training.py [--nav-only | --fetch-only | --both]
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import argparse


class ParallelTrainer:
    def __init__(self):
        self.processes = {}
        self.start_time = datetime.now()
        self.logs_dir = "logs/parallel"
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def start_navigation_training(self):
        """Start low-level navigation training"""
        log_file = f"{self.logs_dir}/nav_training_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        
        print(f"\n{'='*70}")
        print(f"Starting Navigation Training")
        print(f"{'='*70}")
        print(f"Output: {log_file}")
        
        with open(log_file, 'w') as f:
            f.write(f"Navigation Training Started: {self.start_time}\n")
            f.write(f"{'='*70}\n\n")
        
        cmd = [
            sys.executable,
            "src/train_low_level_nav.py"
        ]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
            self.processes['navigation'] = {
                'process': proc,
                'log_file': log_file,
                'cmd': ' '.join(cmd)
            }
            print(f"✓ Navigation training started (PID: {proc.pid})")
            return True
        except Exception as e:
            print(f"✗ Failed to start navigation training: {e}")
            return False
    
    def start_fetch_training(self):
        """Start Fetch arm reaching training"""
        log_file = f"{self.logs_dir}/fetch_arm_training_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        
        print(f"\n{'='*70}")
        print(f"Starting Fetch Arm Reaching Training")
        print(f"{'='*70}")
        print(f"Output: {log_file}")
        
        with open(log_file, 'w') as f:
            f.write(f"Fetch Arm Training Started: {self.start_time}\n")
            f.write(f"{'='*70}\n\n")
        
        cmd = [
            sys.executable,
            "src/arm/train_fetch_arm.py"
        ]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
            self.processes['fetch_arm'] = {
                'process': proc,
                'log_file': log_file,
                'cmd': ' '.join(cmd)
            }
            print(f"✓ Fetch arm training started (PID: {proc.pid})")
            return True
        except Exception as e:
            print(f"✗ Failed to start fetch arm training: {e}")
            return False
    
    def monitor_training(self):
        """Monitor training processes"""
        print(f"\n{'='*70}")
        print(f"Monitoring Training Processes")
        print(f"{'='*70}\n")
        
        while self.processes:
            try:
                # Check status of each process
                active = []
                for name, info in self.processes.items():
                    proc = info['process']
                    if proc.poll() is None:  # Process still running
                        active.append(name)
                        print(f"✓ {name.upper():15} | PID: {proc.pid:6} | Running")
                    else:
                        # Process finished
                        return_code = proc.returncode
                        status = "✓ Completed" if return_code == 0 else f"✗ Failed (code: {return_code})"
                        print(f"{status} | {name.upper():15} | Log: {info['log_file']}")
                
                if not active:
                    break
                
                # Wait before next check
                time.sleep(5)
                print()  # Spacing between updates
                
            except KeyboardInterrupt:
                print("\n\n⚠ Training interrupted by user")
                self.stop_all_training()
                break
            except Exception as e:
                print(f"✗ Error monitoring: {e}")
                break
    
    def stop_all_training(self):
        """Stop all training processes"""
        print(f"\n{'='*70}")
        print(f"Stopping All Training Processes")
        print(f"{'='*70}\n")
        
        for name, info in self.processes.items():
            proc = info['process']
            if proc.poll() is None:
                print(f"Stopping {name}... ", end='', flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                    print("✓")
                except subprocess.TimeoutExpired:
                    print("(force killing)")
                    proc.kill()
    
    def print_summary(self):
        """Print training summary"""
        elapsed = datetime.now() - self.start_time
        print(f"\n{'='*70}")
        print(f"Training Summary")
        print(f"{'='*70}")
        print(f"Total elapsed time: {elapsed}")
        print(f"Logs directory: {self.logs_dir}\n")
        
        for name, info in self.processes.items():
            proc = info['process']
            status = "COMPLETED" if proc.returncode == 0 else f"FAILED (code: {proc.returncode})"
            print(f"  {name.upper():15} | {status}")
            print(f"  Log: {info['log_file']}\n")


def main():
    parser = argparse.ArgumentParser(description="Parallel training launcher")
    parser.add_argument(
        '--mode',
        choices=['both', 'nav-only', 'fetch-only'],
        default='both',
        help='Which trainers to run'
    )
    parser.add_argument(
        '--monitor-only',
        action='store_true',
        help='Only monitor already running processes'
    )
    
    args = parser.parse_args()
    
    trainer = ParallelTrainer()
    
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL RL - PARALLEL TRAINING LAUNCHER")
    print(f"Started: {trainer.start_time}")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Use Ctrl+C to stop all training")
    
    # Start training
    if args.mode in ['both', 'nav-only']:
        trainer.start_navigation_training()
        time.sleep(2)  # Brief delay between starts
    
    if args.mode in ['both', 'fetch-only']:
        trainer.start_fetch_training()
    
    if trainer.processes:
        # Monitor training
        trainer.monitor_training()
        trainer.print_summary()
    else:
        print("✗ No training processes started")


if __name__ == "__main__":
    main()
