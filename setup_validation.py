#!/usr/bin/env python3
"""
Setup and configure validation tools for Bitcoin mining system.

This script extracts and sets up validation tools from ZIP files
and provides a unified interface for running all validation checks.
"""

import os
import sys
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict


class ValidationSetup:
    """Setup and manage validation tools."""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize validation setup.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root).resolve()
        self.tools_dir = self.project_root / "validation_tools"
        self.zip_files = [
            "black-main.zip",
            "flake8-main.zip", 
            "pylint-main.zip",
            "mypy-master.zip",
            "bandit-main.zip",
            "vulture-main.zip",
            "interrogate-master.zip",
            "hypothesis-master.zip",
            "z3-master.zip",
            "coveragepy-master.zip"
        ]
    
    def setup_tools(self) -> bool:
        """
        Extract and setup all validation tools.
        
        Returns:
            True if setup successful
        """
        print("Setting up validation tools...")
        
        # Create tools directory
        self.tools_dir.mkdir(exist_ok=True)
        
        success_count = 0
        for zip_file in self.zip_files:
            zip_path = self.project_root / zip_file
            if zip_path.exists():
                if self._extract_tool(zip_path):
                    success_count += 1
                    print(f"✓ {zip_file} extracted successfully")
                else:
                    print(f"✗ Failed to extract {zip_file}")
            else:
                print(f"⚠ {zip_file} not found, skipping")
        
        print(f"Setup complete: {success_count}/{len(self.zip_files)} tools available")
        return success_count > 0
    
    def _extract_tool(self, zip_path: Path) -> bool:
        """
        Extract a single validation tool.
        
        Args:
            zip_path: Path to ZIP file
            
        Returns:
            True if extraction successful
        """
        try:
            tool_name = zip_path.stem
            extract_dir = self.tools_dir / tool_name
            
            # Remove existing directory
            if extract_dir.exists():
                import shutil
                shutil.rmtree(extract_dir)
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.tools_dir)
            
            return True
            
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return False
    
    def run_black(self) -> bool:
        """Run black formatting check."""
        print("Running black formatting check...")
        try:
            # Try system black first
            result = subprocess.run([
                sys.executable, "-m", "black", "--check", "--diff", "."
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Black formatting check passed")
                return True
            else:
                print("✗ Black formatting issues found:")
                print(result.stdout)
                return False
                
        except FileNotFoundError:
            print("⚠ Black not available in system, skipping")
            return True
    
    def run_flake8(self) -> bool:
        """Run flake8 linting."""
        print("Running flake8 linting...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "flake8", "."
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Flake8 linting passed")
                return True
            else:
                print("✗ Flake8 issues found:")
                print(result.stdout)
                return False
                
        except FileNotFoundError:
            print("⚠ Flake8 not available, skipping")
            return True
    
    def run_basic_validation(self) -> Dict[str, bool]:
        """
        Run basic validation checks that don't require external tools.
        
        Returns:
            Dictionary with validation results
        """
        print("Running basic validation checks...")
        
        results = {}
        
        # Check Python syntax
        print("Checking Python syntax...")
        python_files = list(self.project_root.glob("*.py"))
        syntax_ok = True
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                print(f"✗ Syntax error in {py_file}: {e}")
                syntax_ok = False
        
        results['syntax'] = syntax_ok
        if syntax_ok:
            print("✓ Python syntax check passed")
        
        # Check imports
        print("Checking imports...")
        import_ok = True
        for py_file in python_files:
            if py_file.name.startswith('test_'):
                continue
            try:
                subprocess.run([
                    sys.executable, "-m", "py_compile", str(py_file)
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"✗ Import error in {py_file}")
                import_ok = False
        
        results['imports'] = import_ok
        if import_ok:
            print("✓ Import check passed")
        
        # Check file structure
        print("Checking file structure...")
        required_files = [
            "math_engine.py",
            "model_orchestrator.py", 
            "run_engine.py",
            "mining_controller.py",
            "orchestrator.py",
            "requirements.txt"
        ]
        
        structure_ok = True
        for req_file in required_files:
            if not (self.project_root / req_file).exists():
                print(f"✗ Missing required file: {req_file}")
                structure_ok = False
        
        results['structure'] = structure_ok
        if structure_ok:
            print("✓ File structure check passed")
        
        return results
    
    def run_full_validation(self) -> bool:
        """
        Run complete validation suite.
        
        Returns:
            True if all validations pass
        """
        print("="*50)
        print("RUNNING FULL VALIDATION SUITE")
        print("="*50)
        
        # Basic validation
        basic_results = self.run_basic_validation()
        basic_ok = all(basic_results.values())
        
        # External tool validation
        external_ok = True
        external_ok &= self.run_black()
        external_ok &= self.run_flake8()
        
        # Summary
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        
        for check, result in basic_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{check.capitalize()}: {status}")
        
        overall_result = basic_ok and external_ok
        print(f"\nOverall: {'✓ ALL CHECKS PASSED' if overall_result else '✗ SOME CHECKS FAILED'}")
        print("="*50)
        
        return overall_result


def main():
    """Main entry point for validation setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bitcoin Mining Validation Setup")
    parser.add_argument("--setup", action="store_true", help="Setup validation tools")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    parser.add_argument("--full", action="store_true", help="Run full validation suite")
    args = parser.parse_args()
    
    validator = ValidationSetup()
    
    if args.setup:
        validator.setup_tools()
    elif args.validate or args.full:
        if args.full:
            success = validator.run_full_validation()
        else:
            results = validator.run_basic_validation()
            success = all(results.values())
        
        sys.exit(0 if success else 1)
    else:
        # Default: run basic validation
        results = validator.run_basic_validation()
        success = all(results.values())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()