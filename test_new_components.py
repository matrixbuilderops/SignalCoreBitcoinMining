"""
Test module for the new Bitcoin mining components.

Tests for listener.py, model_caller.py, output_handler.py,
submission_client.py, and block_miner.py.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from listener import BitcoinBlockListener
from model_caller import ModelCaller
from output_handler import OutputHandler, OutputMode
from submission_client import SubmissionClient
from block_miner import BlockMiner, MiningStats


def test_bitcoin_block_listener():
    """Test BitcoinBlockListener functionality."""
    listener = BitcoinBlockListener(verbose=False)

    # Test callback setting
    mock_callback = Mock()
    listener.set_callback(mock_callback)
    assert listener.callback_handler == mock_callback

    # Test without callback should raise error
    listener.callback_handler = None
    with pytest.raises(ValueError):
        listener.start_listening()

    return True


def test_model_caller():
    """Test ModelCaller functionality."""
    caller = ModelCaller(verbose=False)

    # Test math context injection
    test_data = {
        "level": 16000,
        "pre_drift": True,
        "fork_integrity": True,
        "sorrell": 123456,
        "bit_load": 1600000,
    }

    context = caller.inject_math_context(test_data)
    assert "LEVEL 16000" in context
    assert "Sorrell: Knuth(10, 3, 16000) = 123456" in context
    assert "BitLoad: 1600000" in context

    # Test model stats
    stats = caller.get_model_stats()
    assert "total_calls" in stats
    assert "success_rate_percent" in stats
    assert stats["model_name"] == "mixtral:8x7b-instruct-v0.1-q6_K"

    return True


def test_output_handler():
    """Test OutputHandler functionality."""
    handler = OutputHandler(OutputMode.QUIET)

    # Test mode setting
    assert handler.mode == OutputMode.QUIET
    handler.set_mode(OutputMode.VERBOSE)
    assert handler.mode == OutputMode.VERBOSE

    # Test thinking mode
    handler.set_mode(OutputMode.THINKING)
    handler.start_thinking()
    assert handler.thinking_active
    handler.stop_thinking()
    assert not handler.thinking_active

    # Test progress bar creation
    progress = handler.create_progress_bar(25, 100, 20)
    assert "[=====" in progress
    assert "25.0%" in progress

    return True


def test_submission_client():
    """Test SubmissionClient functionality."""
    client = SubmissionClient(verbose=False)

    # Test solution data preparation
    test_validation = {
        "level": 16000,
        "fork_integrity": True,
        "entropy_parity": True,
        "fork_sync": True,
        "sorrell": 123456,
        "pre_stabilizer": "test_hash",
        "post_stabilizer": "test_hash2",
    }

    solution_data = client.prepare_solution_data(test_validation)
    assert solution_data["level"] == 16000
    assert solution_data["fork_integrity"] is True
    assert (
        solution_data["wallet_address"] == "bc1qcmxyhlpm3lf9zvyevqas9n547lywws2e7wvuu1"
    )
    assert "submission_time" in solution_data

    # Test stats
    stats = client.get_submission_stats()
    assert "total_submissions" in stats
    assert "success_rate_percent" in stats
    assert stats["wallet_address"] == "bc1qcmxyhlpm3lf9zvyevqas9n547lywws2e7wvuu1"

    return True


def test_mining_stats():
    """Test MiningStats data structure."""
    stats = MiningStats()

    # Test initial values
    assert stats.blocks_detected == 0
    assert stats.blocks_processed == 0
    assert stats.solutions_generated == 0
    assert stats.successful_submissions == 0

    # Test modifications
    stats.blocks_detected = 5
    stats.successful_submissions = 2
    assert stats.blocks_detected == 5
    assert stats.successful_submissions == 2

    return True


def test_block_miner_initialization():
    """Test BlockMiner initialization."""
    miner = BlockMiner(verbose=False, thinking_mode=False)

    # Test component initialization
    assert miner.listener is not None
    assert miner.model_caller is not None
    assert miner.submission_client is not None
    assert miner.output is not None

    # Test stats initialization
    assert isinstance(miner.stats, MiningStats)
    assert miner.stats.blocks_detected == 0

    # Test initial state
    assert not miner.running

    return True


def test_block_miner_stats():
    """Test BlockMiner statistics functionality."""
    miner = BlockMiner(verbose=False)
    miner.stats.start_time = time.time() - 60  # 1 minute ago
    miner.stats.blocks_processed = 10
    miner.stats.successful_submissions = 3

    stats = miner.get_mining_stats()
    assert stats["blocks_processed"] == 10
    assert stats["successful_submissions"] == 3
    assert stats["success_rate_percent"] == 30.0
    assert stats["runtime_minutes"] > 0.9  # Should be close to 1 minute

    return True


@patch("listener.listen_for_blocks")
def test_block_miner_callback_setup(mock_listen):
    """Test that BlockMiner properly sets up block callbacks."""
    miner = BlockMiner(verbose=False)

    # Mock the readiness check to return True
    with patch.object(miner, "_verify_system_readiness", return_value=True):
        # Start should set up the callback and call listen_for_blocks
        try:
            miner.start()
        except Exception:
            pass  # Expected since we're mocking

    # Verify that listen_for_blocks was called
    mock_listen.assert_called_once()

    return True


def test_model_caller_analysis():
    """Test ModelCaller analysis functionality."""
    caller = ModelCaller(verbose=False)

    test_validation = {
        "level": 16000,
        "fork_integrity": True,
        "entropy_parity": True,
        "fork_sync": True,
        "pre_drift": True,
        "post_drift": True,
        "sorrell": 123456,
        "fork_cluster": 234567,
        "over_recursion": 345678,
    }

    # Mock the AI response to avoid calling actual Ollama
    with patch.object(
        caller, "call_model_with_math", return_value="PROCEED: All validations pass"
    ):
        analysis = caller.analyze_mining_opportunity(test_validation, "test_hash")

    assert analysis["recommendation"] == "PROCEED"
    assert analysis["validation_score"] == 100.0
    assert analysis["critical_checks_passed"] == 5
    assert analysis["should_proceed"] is True
    assert analysis["level"] == 16000

    return True


def test_output_handler_modes():
    """Test OutputHandler different output modes."""
    handler = OutputHandler(OutputMode.VERBOSE)

    # Test verbose mode - should output
    assert handler._should_output("INFO") is True
    assert handler._should_output("ERROR") is True

    # Test quiet mode - should not output
    handler.set_mode(OutputMode.QUIET)
    assert handler._should_output("INFO") is False
    assert handler._should_output("ERROR") is False

    # Test thinking mode - only errors and success
    handler.set_mode(OutputMode.THINKING)
    assert handler._should_output("INFO") is False
    assert handler._should_output("ERROR") is True
    assert handler._should_output("SUCCESS") is True

    # Test error only mode
    handler.set_mode(OutputMode.ERROR_ONLY)
    assert handler._should_output("INFO") is False
    assert handler._should_output("ERROR") is True
    assert handler._should_output("WARNING") is True

    return True


def test_submission_client_success_rate():
    """Test SubmissionClient success rate calculation."""
    client = SubmissionClient(verbose=False)

    # No submissions
    assert client.get_success_rate() == 0.0

    # Simulate submissions
    client.submissions_attempted = 10
    client.submissions_successful = 3
    assert client.get_success_rate() == 30.0

    # All successful
    client.submissions_successful = 10
    assert client.get_success_rate() == 100.0

    return True


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_bitcoin_block_listener,
        test_model_caller,
        test_output_handler,
        test_submission_client,
        test_mining_stats,
        test_block_miner_initialization,
        test_block_miner_stats,
        test_block_miner_callback_setup,
        test_model_caller_analysis,
        test_output_handler_modes,
        test_submission_client_success_rate,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                print(f"✓ {test.__name__}")
                passed += 1
            else:
                print(f"✗ {test.__name__}")
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")
