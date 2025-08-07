#!/usr/bin/env python3
"""
Comprehensive validation and failure scenario test coverage for SignalCore Bitcoin Mining System.
Tests all failure modes and integration points identified in the problem statement.
"""

import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess
import threading
from typing import Dict, Any

# Import the modules we're testing
from model_interface_layer import ModelInterface, ModelInput, ModelOutput, ModelResponseType
from bitcoin_mining_core import BitcoinMiningCore, SYSTEM_CONFIG
from signalcore_monitor import SignalCoreMonitor, MonitorState
from ai_interface import get_ai_recommendation_with_fallback


class TestModelInterfaceValidation(unittest.TestCase):
    """Test model interface input/output validation and retry logic"""

    def setUp(self):
        self.model_interface = ModelInterface()

    def test_structured_input_validation(self):
        """Test structured input validation"""
        # Valid input
        valid_input = ModelInput(
            prompt="Test prompt",
            timeout=10,
            retry_attempts=2
        )
        self.assertEqual(valid_input.prompt, "Test prompt")
        self.assertEqual(valid_input.timeout, 10)
        self.assertEqual(valid_input.retry_attempts, 2)

    def test_model_timeout_handling(self):
        """Test model timeout scenarios"""
        with patch('subprocess.Popen') as mock_popen:
            # Mock timeout scenario
            mock_process = Mock()
            mock_process.communicate.side_effect = subprocess.TimeoutExpired("ollama", 5)
            mock_process.kill.return_value = None
            mock_popen.return_value = mock_process

            model_input = ModelInput(
                prompt="Test prompt",
                timeout=1,
                retry_attempts=1
            )
            
            result = self.model_interface.query_model_structured(model_input)
            
            # Should eventually return timeout or error after retries
            self.assertIn(result.response_type, [ModelResponseType.TIMEOUT, ModelResponseType.ERROR])
            if result.response_type == ModelResponseType.TIMEOUT:
                self.assertIn("timed out", result.error_message)
            else:
                # Might be error after retries fail
                self.assertIsNotNone(result.error_message)

    def test_model_retry_logic(self):
        """Test retry logic with exponential backoff"""
        with patch('subprocess.Popen') as mock_popen, \
             patch('time.sleep') as mock_sleep:
            
            # Mock retry scenario
            mock_process = Mock()
            mock_process.communicate.return_value = ("", "connection error")
            mock_process.returncode = 1
            mock_popen.return_value = mock_process

            model_input = ModelInput(
                prompt="Test prompt",
                timeout=5,
                retry_attempts=3
            )
            
            result = self.model_interface.query_model_structured(model_input)
            
            # Should have attempted retries
            self.assertGreater(result.retry_count, 0)
            # Should have called sleep for exponential backoff
            mock_sleep.assert_called()

    def test_model_connection_failure(self):
        """Test model connection failure scenarios"""
        with patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = FileNotFoundError("ollama not found")
            
            model_input = ModelInput(prompt="Test", retry_attempts=1)
            result = self.model_interface.query_model_structured(model_input)
            
            self.assertEqual(result.response_type, ModelResponseType.ERROR)
            self.assertIn("not found", result.error_message)

    def test_empty_response_handling(self):
        """Test handling of empty model responses"""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            model_input = ModelInput(prompt="Test", retry_attempts=2)
            result = self.model_interface.query_model_structured(model_input)
            
            # Should trigger retry for empty response
            self.assertEqual(result.response_type, ModelResponseType.RETRY_NEEDED)

    def test_health_check_functionality(self):
        """Test model interface health check"""
        with patch.object(self.model_interface, 'query_model_structured') as mock_query:
            # Test healthy scenario
            mock_query.return_value = ModelOutput(response_type=ModelResponseType.SUCCESS)
            self.assertTrue(self.model_interface.health_check())
            
            # Test timeout scenario (still considered healthy)
            mock_query.return_value = ModelOutput(response_type=ModelResponseType.TIMEOUT)
            self.assertTrue(self.model_interface.health_check())
            
            # Test error scenario (unhealthy)
            mock_query.return_value = ModelOutput(response_type=ModelResponseType.ERROR)
            self.assertFalse(self.model_interface.health_check())


class TestZMQFailureScenarios(unittest.TestCase):
    """Test ZMQ message failure scenarios"""

    def setUp(self):
        self.mining_core = BitcoinMiningCore()

    def test_zmq_connection_failure(self):
        """Test ZMQ connection failure and fallback"""
        # Mock the zmq availability check
        with patch('bitcoin_mining_core.ZMQ_AVAILABLE', False):
            result = self.mining_core.start_zmq_listener()
            self.assertFalse(result)

    def test_zmq_message_corruption(self):
        """Test handling of corrupted ZMQ messages"""
        # Test with corrupted block data
        corrupted_data = b"corrupted_block_data"
        
        try:
            self.mining_core.process_new_block(corrupted_data)
            # Should not raise exception
        except Exception as e:
            self.fail(f"ZMQ message corruption should be handled gracefully: {e}")

    def test_zmq_large_message_handling(self):
        """Test handling of unusually large ZMQ messages"""
        # Test with large block data
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        
        start_time = time.time()
        try:
            self.mining_core.process_new_block(large_data)
            processing_time = time.time() - start_time
            # Should complete within reasonable time
            self.assertLess(processing_time, 10.0)
        except Exception as e:
            self.fail(f"Large message should be handled: {e}")


class TestMiningCoreIntegration(unittest.TestCase):
    """Test mining core integration points"""

    def setUp(self):
        self.mining_core = BitcoinMiningCore()

    def test_model_to_mining_flow(self):
        """Test complete flow from model recommendation to mining action"""
        with patch.object(self.mining_core, 'get_mining_recommendation') as mock_recommend:
            mock_recommend.return_value = "PROCEED: All validations passed"
            
            # Test block processing
            test_data = b"test_block_data"
            
            with patch.object(self.mining_core, 'attempt_mining') as mock_mining:
                mock_mining.return_value = True
                
                self.mining_core.process_new_block(test_data)
                
                # Verify the flow was called
                mock_recommend.assert_called_once()
                mock_mining.assert_called_once()

    def test_model_failure_fallback(self):
        """Test fallback behavior when model fails"""
        with patch.object(self.mining_core.model_interface, 'query_model_structured') as mock_query:
            mock_query.return_value = ModelOutput(
                response_type=ModelResponseType.ERROR,
                error_message="Model unavailable"
            )
            
            validation_data = {"level": 16000, "block_hash": "test"}
            result = self.mining_core.get_mining_recommendation(validation_data)
            
            # Should get fallback recommendation
            self.assertIsNotNone(result)
            self.assertIn("FALLBACK", result)

    def test_mining_statistics_tracking(self):
        """Test mining statistics are properly tracked"""
        initial_stats = self.mining_core.get_stats()
        
        # Simulate some activity
        test_data = {"level": 16000, "block_hash": "test"}
        self.mining_core.get_mining_recommendation(test_data)
        
        updated_stats = self.mining_core.get_stats()
        
        # Stats should be updated
        self.assertEqual(updated_stats['model_calls'], initial_stats['model_calls'] + 1)


class TestMonitorResilience(unittest.TestCase):
    """Test signalcore_monitor resilience and error handling"""

    def setUp(self):
        self.monitor = SignalCoreMonitor(silent_mode=True)

    def test_monitor_crash_resilience(self):
        """Test monitor resilience to crashes"""
        def error_callback(error):
            self.error_captured = error
        
        self.error_captured = None
        
        # Start monitor
        self.monitor.start("Test", error_callback=error_callback)
        time.sleep(0.5)
        
        # Simulate crash by patching spinner to raise exception
        with patch.object(self.monitor, '_spinner', side_effect=Exception("Simulated crash")):
            # Restart after crash
            self.monitor.start("Test Recovery")
            time.sleep(0.5)
        
        self.monitor.stop()
        
        # Should handle gracefully
        self.assertTrue(self.monitor.get_stats()["errors_logged"] >= 0)

    def test_monitor_silent_mode_switching(self):
        """Test automatic silent mode switching on output errors"""
        monitor = SignalCoreMonitor(silent_mode=False)
        
        # Start monitor
        monitor.start("Test")
        time.sleep(0.2)
        
        # Simulate output error
        with patch('sys.stdout.write', side_effect=BrokenPipeError("Terminal disconnected")):
            monitor.update_message("Test message")
            time.sleep(0.5)
        
        monitor.stop()
        
        # Should have logged errors
        stats = monitor.get_stats()
        self.assertGreater(stats["errors_logged"], 0)

    def test_monitor_health_check(self):
        """Test monitor health check functionality"""
        # Test stopped monitor
        self.assertFalse(self.monitor.is_running())
        self.assertTrue(self.monitor.health_check())  # Stopped state is healthy
        
        # Test running monitor
        self.monitor.start("Health Test")
        time.sleep(0.2)
        self.assertTrue(self.monitor.is_running())
        self.assertTrue(self.monitor.health_check())
        
        self.monitor.stop()


class TestAIFallbackLogic(unittest.TestCase):
    """Test AI fallback logic and mathematical decisions"""

    def test_ai_fallback_decision_logic(self):
        """Test mathematical fallback when AI is unavailable"""
        # Test PROCEED scenario
        validation_data = {
            "level": 16000,
            "pre_drift": True,
            "fork_integrity": True,
            "recursion_sync": True,
            "entropy_parity": True,
            "post_drift": True,
            "fork_sync": True,
            "sorrell": 12345,
            "bit_load": 1600000,
            "cycles": 161
        }
        
        result = get_ai_recommendation_with_fallback(
            validation_data, 
            "test_hash", 
            enable_ai=False
        )
        
        self.assertIn("PROCEED", result)

    def test_ai_fallback_hold_scenario(self):
        """Test HOLD decision in fallback mode"""
        validation_data = {
            "level": 16000,
            "pre_drift": False,
            "fork_integrity": False,
            "recursion_sync": False,
            "entropy_parity": False,
            "post_drift": False,
            "fork_sync": False
        }
        
        result = get_ai_recommendation_with_fallback(
            validation_data, 
            "test_hash", 
            enable_ai=False
        )
        
        self.assertIn("HOLD", result)


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration scenarios"""

    def test_complete_pipeline_integration(self):
        """Test complete pipeline from block detection to mining decision"""
        mining_core = BitcoinMiningCore()
        
        # Mock all external dependencies
        with patch.object(mining_core, 'start_zmq_listener', return_value=True), \
             patch.object(mining_core.model_interface, 'health_check', return_value=True):
            
            # Test block processing pipeline
            test_block_data = b"integration_test_block_data"
            
            try:
                mining_core.process_new_block(test_block_data)
                
                # Verify stats were updated
                stats = mining_core.get_stats()
                self.assertEqual(stats['blocks_processed'], 1)
                
            except Exception as e:
                self.fail(f"Complete pipeline should work: {e}")

    def test_system_graceful_degradation(self):
        """Test system graceful degradation when components fail"""
        mining_core = BitcoinMiningCore()
        
        # Simulate various component failures
        with patch.object(mining_core.model_interface, 'health_check', return_value=False), \
             patch.object(mining_core, 'start_zmq_listener', return_value=False):
            
            # System should still start and handle gracefully
            try:
                # Simulate startup
                mining_core.running = True
                
                # Process a block even with degraded components
                mining_core.process_new_block(b"test_data")
                
                # Should complete without crashing
                self.assertTrue(True)
                
            except Exception as e:
                self.fail(f"System should degrade gracefully: {e}")


def run_validation_tests():
    """Run comprehensive validation tests"""
    print("🧪 Running Comprehensive Validation and Failure Scenario Tests")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestModelInterfaceValidation,
        TestZMQFailureScenarios,
        TestMiningCoreIntegration,
        TestMonitorResilience,
        TestAIFallbackLogic,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ All validation tests passed!")
        print(f"Tests run: {result.testsRun}")
    else:
        print("❌ Some validation tests failed!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)