#!/usr/bin/env python3
"""
Test script for Bitcoin Mining Engine
Validates all components work correctly
"""

import sys


def test_math_module() -> bool:
    """Test mathematical processing module"""
    print("Testing math module...")
    try:
        from math_module import process_block_with_math, knuth_algorithm

        test_data = b"test_block_data_for_validation"
        processed, results = process_block_with_math(test_data, 16000)

        # Validate results structure
        required_keys = [
            "level",
            "fork_integrity",
            "entropy_parity",
            "fork_sync",
            "sorrell",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

        # Validate Knuth algorithm
        knuth_result = knuth_algorithm(10, 3, 16000)  # Use Level 16000
        assert knuth_result > 0, "Knuth algorithm should return positive value"

        print("  ✓ Math module passed")
        return True
    except Exception as e:
        print(f"  ✗ Math module failed: {e}")
        return False


def test_ai_interface() -> bool:
    """Test AI interface module"""
    print("Testing AI interface...")
    try:
        from ai_interface import call_ai_model, extract_recommendation

        mock_validation = {
            "level": 16000,
            "fork_integrity": True,
            "entropy_parity": True,
            "fork_sync": True,
            "sorrell": 123,
        }

        response = call_ai_model(mock_validation, "test_hash")
        recommendation = extract_recommendation(response)

        # Should handle errors gracefully
        valid_recs = ["PROCEED", "HOLD", "RETRY", "ERROR"]
        assert recommendation in valid_recs, f"Invalid recommendation: {recommendation}"

        print("  ✓ AI interface passed")
        return True
    except Exception as e:
        print(f"  ✗ AI interface failed: {e}")
        return False


def test_mining_controller() -> bool:
    """Test mining controller module"""
    print("Testing mining controller...")
    try:
        from mining_controller import validate_solution, BITCOIN_ADDRESS

        # Test with valid solution
        valid_solution = {
            "fork_integrity": True,
            "entropy_parity": True,
            "fork_sync": True,
        }

        # Test with invalid solution
        invalid_solution = {
            "fork_integrity": False,
            "entropy_parity": True,
            "fork_sync": True,
        }

        assert validate_solution(valid_solution) is True, "Valid solution should pass"
        assert (
            validate_solution(invalid_solution) is False
        ), "Invalid solution should fail"
        assert len(BITCOIN_ADDRESS) > 20, "Bitcoin address should be valid length"

        print("  ✓ Mining controller passed")
        return True
    except Exception as e:
        print(f"  ✗ Mining controller failed: {e}")
        return False


def test_block_listener() -> bool:
    """Test block listener module"""
    print("Testing block listener...")
    try:
        from block_listener import create_mock_block_data

        test_hash = "abcdef1234567890" * 4  # 64 char hash
        mock_data = create_mock_block_data(test_hash)

        assert len(mock_data) > 32, "Mock block data should be substantial"
        assert isinstance(mock_data, bytes), "Mock data should be bytes"

        print("  ✓ Block listener passed")
        return True
    except Exception as e:
        print(f"  ✗ Block listener failed: {e}")
        return False


def test_orchestrator() -> bool:
    """Test orchestrator module"""
    print("Testing orchestrator...")
    try:
        from orchestrator import BitcoinMiningOrchestrator

        orchestrator = BitcoinMiningOrchestrator(verbose=False, ai_enabled=False)

        # Test block processing without actual mining
        topic = b"hashblock"
        message = b"test_block_hash_for_validation_12345678"

        # This should not raise exceptions
        orchestrator.on_block_received(topic, message)

        assert orchestrator.blocks_processed == 1, "Should have processed one block"

        print("  ✓ Orchestrator passed")
        return True
    except Exception as e:
        print(f"  ✗ Orchestrator failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Running Bitcoin Mining Engine Tests")
    print("=" * 50)

    tests = [
        test_math_module,
        test_ai_interface,
        test_mining_controller,
        test_block_listener,
        test_orchestrator,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed! System is ready for use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
