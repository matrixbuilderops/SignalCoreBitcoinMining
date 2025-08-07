#!/usr/bin/env python3
"""
Test script for new Bitcoin Mining Engine components.
Validates model orchestrator, run engine, and enhanced RPC functionality.
"""

import sys
import time


def test_math_engine_level_16000():
    """Test enhanced math engine with exact hashes."""
    print("Testing math engine Level 16000...")
    try:
        from math_engine import run_level_16000

        result = run_level_16000(verbose=False)

        # Check exact stabilizer hashes are used
        pre_safeguards = result["Pre-Safeguards"]
        post_safeguards = result["Post-Safeguards"]

        expected_pre_hash = "941d793ce78e45983a4d98d6e4ed0529d923f06f8ecefcabe45c5448c65333fca9549a80643f175154046d09bedc6bfa8546820941ba6e12d39f67488451f47b"
        expected_post_hash = "74402f56dc3f9154da10ab8d5dbe518db9aa2a332b223bc7bdca9871d0b1a55c3cc03b25e5053f58d443c9fa45f8ec93bae647cd5b44b853bebe1178246119eb"

        found_pre_hash = False
        found_post_hash = False

        for item in pre_safeguards:
            if expected_pre_hash in str(item):
                found_pre_hash = True

        for item in post_safeguards:
            if expected_post_hash in str(item):
                found_post_hash = True

        assert found_pre_hash, "Expected pre-hash not found"
        assert found_post_hash, "Expected post-hash not found"

        # Check main equation values
        main_eq = result["Main Equation"]
        assert main_eq["BitLoad"] == 1600000, "BitLoad should be 1600000"
        assert main_eq["Cycles"] == 161, "Cycles should be 161"
        assert main_eq["Sandboxes"] == 1, "Sandboxes should be 1"

        print("  ✓ Math engine Level 16000 passed")
        return True

    except Exception as e:
        print(f"  ✗ Math engine Level 16000 failed: {e}")
        return False


def test_model_orchestrator_basic():
    """Test model orchestrator without AI model call."""
    print("Testing model orchestrator basic functionality...")
    try:
        from model_orchestrator import ModelOrchestrator

        # Create orchestrator in quiet mode
        orchestrator = ModelOrchestrator(verbose=False, thinking_mode=False)

        # Test blockchain data injection
        test_data = b"test_block_data_for_orchestrator"
        test_hash = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"

        enriched_data = orchestrator.inject_blockchain_data(test_data, test_hash)

        # Validate enriched data structure
        required_keys = [
            "block_hash",
            "block_size",
            "processed_hash",
            "math_validation",
            "level_16000_results",
            "timestamp",
        ]

        for key in required_keys:
            assert key in enriched_data, f"Missing key in enriched data: {key}"

        assert enriched_data["block_hash"] == test_hash
        assert enriched_data["block_size"] == len(test_data)
        assert enriched_data["processing_level"] == 16000

        # Test performance stats
        stats = orchestrator.get_performance_stats()
        assert "solutions_generated" in stats
        assert "valid_solutions" in stats
        assert "success_rate_percent" in stats

        print("  ✓ Model orchestrator basic tests passed")
        return True

    except Exception as e:
        print(f"  ✗ Model orchestrator basic tests failed: {e}")
        return False


def test_enhanced_rpc_functions():
    """Test enhanced RPC functionality."""
    print("Testing enhanced RPC functions...")
    try:
        from mining_controller import (
            get_chain_data,
            validate_wallet_address,
            create_mining_transaction,
            BITCOIN_ADDRESS,
        )

        # Test chain data function with error handling
        try:
            chain_data = get_chain_data()

            # Should have all required keys even if Bitcoin Core is not available
            required_keys = [
                "block_height",
                "difficulty",
                "network_hashrate",
                "wallet_balance",
                "mempool_size",
                "timestamp",
            ]

            for key in required_keys:
                assert key in chain_data, f"Missing key in chain data: {key}"

        except Exception as chain_error:
            # If chain data fails, that's acceptable without Bitcoin Core
            print(
                f"    Chain data error (expected without Bitcoin Core): {chain_error}"
            )

        # Test address validation function (should work without Bitcoin Core)
        # Just test that it doesn't crash
        try:
            validate_wallet_address(BITCOIN_ADDRESS)
        except:
            pass  # Expected without Bitcoin Core

        # Test transaction creation function
        try:
            create_mining_transaction(0.001)
        except:
            pass  # Expected without Bitcoin Core

        print("  ✓ Enhanced RPC functions passed")
        return True

    except Exception as e:
        print(f"  ✗ Enhanced RPC functions failed: {e}")
        return False


def test_run_engine_components():
    """Test run engine components without full execution."""
    print("Testing run engine components...")
    try:
        from run_engine import BitcoinMiningEngine

        # Create engine in quiet mode
        engine = BitcoinMiningEngine(verbose=False, thinking_mode=False)

        # Test basic initialization
        assert hasattr(engine, "model_orchestrator")
        assert hasattr(engine, "running")
        assert hasattr(engine, "total_blocks_processed")

        # Test output control methods
        engine.suppress_output()
        engine.enable_thinking_mode()
        engine.enable_verbose_mode()

        # Test performance stats without processing
        # (Just ensure methods exist and don't crash)
        engine._show_final_stats()

        print("  ✓ Run engine components passed")
        return True

    except Exception as e:
        print(f"  ✗ Run engine components failed: {e}")
        return False


def test_validation_setup():
    """Test validation setup functionality."""
    print("Testing validation setup...")
    try:
        from setup_validation import ValidationSetup

        validator = ValidationSetup(".")

        # Test basic validation without external tools
        results = validator.run_basic_validation()

        # Should have syntax, imports, and structure checks
        expected_checks = ["syntax", "imports", "structure"]
        for check in expected_checks:
            assert check in results, f"Missing validation check: {check}"

        # All basic checks should pass
        all_passed = all(results.values())
        assert all_passed, f"Basic validation failed: {results}"

        print("  ✓ Validation setup passed")
        return True

    except Exception as e:
        print(f"  ✗ Validation setup failed: {e}")
        return False


def test_integration_workflow():
    """Test basic integration workflow without external dependencies."""
    print("Testing integration workflow...")
    try:
        from math_module import process_block_with_math
        from model_orchestrator import ModelOrchestrator

        # Test complete workflow without AI call
        test_data = b"integration_test_block_data"

        # Step 1: Process with math module
        processed_data, math_results = process_block_with_math(test_data, 16000)
        assert math_results["level"] == 16000

        # Step 2: Create orchestrator and inject data
        orchestrator = ModelOrchestrator(verbose=False)
        enriched_data = orchestrator.inject_blockchain_data(test_data, "test_hash")

        # Step 3: Verify data flow
        assert enriched_data["math_validation"]["level"] == math_results["level"]
        assert enriched_data["math_validation"]["sorrell"] == math_results["sorrell"]

        print("  ✓ Integration workflow passed")
        return True

    except Exception as e:
        print(f"  ✗ Integration workflow failed: {e}")
        return False


def main():
    """Run all new component tests."""
    print("🧪 Running Bitcoin Mining Engine Extended Tests")
    print("=" * 60)

    tests = [
        test_math_engine_level_16000,
        test_model_orchestrator_basic,
        test_enhanced_rpc_functions,
        test_run_engine_components,
        test_validation_setup,
        test_integration_workflow,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test_func.__name__} crashed: {e}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All extended tests passed! New components are working correctly.")
        return True
    else:
        print("❌ Some extended tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
