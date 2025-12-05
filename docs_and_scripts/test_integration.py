"""
Integration Test Suite
Validates all components work together: environments, trainers, Docker setup
"""

import subprocess
import sys
import os
from pathlib import Path
import time


class TestSuite:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.workspace = Path(__file__).parent
        
    def print_header(self, title):
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print(f"{'='*70}\n")
    
    def print_test(self, name, status, details=""):
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:50} {status_str:8}")
        if details:
            print(f"  {details}")
        
        if status:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
    
    def test_env_imports(self):
        """Test all environment imports"""
        self.print_header("Testing Environment Imports")
        
        tests = [
            ("SimpleNavigationEnv", "from src.simple_navigation_env import SimpleNavigationEnv"),
            ("FetchNavigationEnv", "from src.arm.fetch_navigation_env import FetchNavigationEnv"),
            ("SkillExecutor", "from src.skill_executor import SkillExecutor, NavigationSkill, FetchSkill"),
        ]
        
        for name, import_stmt in tests:
            try:
                exec(import_stmt)
                self.print_test(name, True)
            except ImportError as e:
                self.print_test(name, False, str(e))
            except Exception as e:
                self.print_test(name, False, f"Unexpected error: {e}")
    
    def test_env_initialization(self):
        """Test environment creation and basic stepping"""
        self.print_header("Testing Environment Initialization")
        
        # Test Navigation Environment
        try:
            from src.simple_navigation_env import SimpleNavigationEnv
            env = SimpleNavigationEnv()
            obs, _ = env.reset()
            
            # Check observation shape
            obs_valid = obs.shape == (2,) and not any(np.isnan(obs))
            
            # Step once
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step_valid = obs.shape == (2,)
            
            env.close()
            self.print_test("Navigation Env Init", obs_valid and step_valid)
        except Exception as e:
            self.print_test("Navigation Env Init", False, str(e))
        
        # Test Fetch Environment
        try:
            from src.arm.fetch_navigation_env import FetchNavigationEnv
            env = FetchNavigationEnv()
            obs, _ = env.reset()
            
            obs_valid = obs.shape == (2,) and not any(np.isnan(obs))
            
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step_valid = obs.shape == (2,)
            
            env.close()
            self.print_test("Fetch Env Init", obs_valid and step_valid)
        except Exception as e:
            self.print_test("Fetch Env Init", False, str(e))
    
    def test_model_paths(self):
        """Check if trained models exist"""
        self.print_header("Testing Model Availability")
        
        models = {
            "Navigation": "models/lowlevel_curriculum_250k.zip",
            "Fetch": "models/fetch/fetch_nav_250k_final.zip",
        }
        
        for name, path in models.items():
            full_path = self.workspace / path
            exists = full_path.exists()
            self.print_test(f"{name} Model", exists, f"Path: {path}")
    
    def test_parallel_launcher(self):
        """Test parallel launcher script existence and syntax"""
        self.print_header("Testing Parallel Launcher")
        
        launcher = self.workspace / "launch_parallel_training.py"
        
        # Check file exists
        exists = launcher.exists()
        self.print_test("Launcher Script Exists", exists)
        
        if exists:
            # Check syntax
            try:
                compile(open(launcher).read(), str(launcher), 'exec')
                self.print_test("Launcher Syntax Valid", True)
            except SyntaxError as e:
                self.print_test("Launcher Syntax Valid", False, str(e))
    
    def test_docker_files(self):
        """Check Docker setup files"""
        self.print_header("Testing Docker Setup")
        
        docker_files = {
            "Dockerfile": "Dockerfile",
            "docker-compose.yml": "docker-compose.yml",
            "DOCKER.md": "DOCKER.md",
        }
        
        for name, path in docker_files.items():
            full_path = self.workspace / path
            exists = full_path.exists()
            self.print_test(name, exists)
    
    def test_documentation(self):
        """Check documentation files"""
        self.print_header("Testing Documentation")
        
        docs = {
            "AI Instructions": ".github/copilot-instructions.md",
            "Implementation Summary": "IMPLEMENTATION_SUMMARY.md",
        }
        
        for name, path in docs.items():
            full_path = self.workspace / path
            exists = full_path.exists()
            
            if exists:
                size = full_path.stat().st_size
                self.print_test(name, exists, f"({size:,} bytes)")
            else:
                self.print_test(name, False)
    
    def test_env_checker(self):
        """Run environment checker on both envs (if sb3 available)"""
        self.print_header("Testing Environment Compliance (SB3 Checker)")
        
        try:
            from stable_baselines3.common.env_checker import check_env
            import numpy as np
            from src.simple_navigation_env import SimpleNavigationEnv
            
            try:
                env = SimpleNavigationEnv()
                check_env(env)
                env.close()
                self.print_test("Navigation Env Compliance", True)
            except Exception as e:
                self.print_test("Navigation Env Compliance", False, str(e)[:50])
            
            try:
                from src.arm.fetch_navigation_env import FetchNavigationEnv
                env = FetchNavigationEnv()
                check_env(env)
                env.close()
                self.print_test("Fetch Env Compliance", True)
            except Exception as e:
                self.print_test("Fetch Env Compliance", False, str(e)[:50])
                
        except ImportError:
            self.print_test("Environment Checker", False, "stable-baselines3 not available")
    
    def test_quick_skill_sequence(self):
        """Quick test of skill executor if models exist"""
        self.print_header("Testing Skill Executor (Requires Trained Models)")
        
        nav_model = self.workspace / "models/lowlevel_curriculum_250k.zip"
        fetch_model = self.workspace / "models/fetch/fetch_nav_250k_final.zip"
        
        if not nav_model.exists() or not fetch_model.exists():
            self.print_test("Skill Executor Test", False, "Models not trained yet")
            return
        
        try:
            from src.skill_executor import SkillExecutor, NavigationSkill, FetchSkill
            from src.simple_navigation_env import SimpleNavigationEnv
            
            executor = SkillExecutor()
            executor.add_skill(NavigationSkill(str(nav_model)))
            
            # Quick test (just 1 step)
            env = SimpleNavigationEnv()
            obs, _ = env.reset()
            action = executor.skills[0].step()
            obs, _, _, _, _ = env.step(action)
            env.close()
            
            self.print_test("Skill Executor", True)
        except Exception as e:
            self.print_test("Skill Executor", False, str(e)[:50])
    
    def print_summary(self):
        """Print test summary"""
        total = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total * 100) if total > 0 else 0
        
        self.print_header("Test Summary")
        
        print(f"Total Tests:  {total}")
        print(f"Passed:       {self.tests_passed} ({'✓' * self.tests_passed})")
        print(f"Failed:       {self.tests_failed} ({'✗' * self.tests_failed})")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\n" + "="*70)
        
        if self.tests_failed == 0:
            print("✓✓✓ ALL TESTS PASSED ✓✓✓")
            return True
        else:
            print("⚠ SOME TESTS FAILED - CHECK ABOVE FOR DETAILS")
            return False


def main():
    print("\n" + "="*70)
    print("HRL Integration Test Suite".center(70))
    print("="*70)
    
    # Check numpy early
    try:
        import numpy as np
    except ImportError:
        print("✗ NumPy not installed - required for tests")
        sys.exit(1)
    
    suite = TestSuite()
    
    # Run tests
    suite.test_env_imports()
    suite.test_env_initialization()
    suite.test_model_paths()
    suite.test_parallel_launcher()
    suite.test_docker_files()
    suite.test_documentation()
    suite.test_env_checker()
    suite.test_quick_skill_sequence()
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
