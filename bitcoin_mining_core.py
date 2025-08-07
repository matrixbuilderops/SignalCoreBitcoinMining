# Core mining orchestration script - entrypoint

import logging
import time
from typing import Dict, Any, Optional
from model_interface_layer import ModelInterface, ModelInput, ModelOutput, ModelResponseType
from ai_interface import get_ai_recommendation_with_fallback

# Import ZMQ with fallback handling
try:
    from zmq_listener import ZMQListener, default_callback
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("[WARNING] ZMQ not available, using fallback mode")

import threading


# System configuration with defaults
SYSTEM_CONFIG = {
    'zmq_endpoints': {
        "hashblock": "tcp://127.0.0.1:28335",
        "rawblock": "tcp://127.0.0.1:28333",
        "hashtx": "tcp://127.0.0.1:28334",
        "rawtx": "tcp://127.0.0.1:28332"
    },
    'recursion_level': 16000,
    'math_equation': 'Knuth(10, 3, 16000)',
    'enable_ai': True,
    'verbose': True
}


def log_event(message: str, level: str = "INFO") -> None:
    """Log system events with proper formatting"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if SYSTEM_CONFIG.get('verbose', True):
        print(f"[{timestamp}] [{level}] {message}")
    
    # Also log to Python logging system
    if level == "ERROR":
        logging.error(message)
    elif level == "WARNING":
        logging.warning(message)
    else:
        logging.info(message)


class BitcoinMiningCore:
    """Enhanced Bitcoin mining core with proper model integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or SYSTEM_CONFIG
        self.model_interface = ModelInterface()
        self.zmq_listener = None
        self.running = False
        self.stats = {
            'blocks_processed': 0,
            'model_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'mining_attempts': 0
        }
    
    def start_zmq_listener(self) -> bool:
        """Start ZMQ listener with proper error handling"""
        if not ZMQ_AVAILABLE:
            log_event("[ZMQ] ZMQ not available, using fallback polling mode", "WARNING")
            return False
            
        try:
            def block_callback(topic: str, msg: bytes) -> None:
                if topic == "hashblock" or topic == "rawblock":
                    self.process_new_block(msg)
            
            self.zmq_listener = ZMQListener(
                self.config['zmq_endpoints'], 
                block_callback
            )
            
            # Start in a separate thread
            zmq_thread = threading.Thread(target=self.zmq_listener.start, daemon=True)
            zmq_thread.start()
            
            log_event("[ZMQ] Listener started successfully")
            return True
            
        except Exception as e:
            log_event(f"[ZMQ] Failed to start listener: {e}", "WARNING")
            return False
    
    def process_new_block(self, block_data: bytes) -> None:
        """Process new block data through the mining pipeline"""
        try:
            self.stats['blocks_processed'] += 1
            log_event(f"[BLOCK] Processing new block ({len(block_data)} bytes)")
            
            # Step 1: Prepare model input with validation data
            model_input_data = self.prepare_model_input(block_data)
            
            # Step 2: Get AI recommendation with fallback
            recommendation = self.get_mining_recommendation(model_input_data)
            
            # Step 3: Act on recommendation
            if recommendation and "PROCEED" in recommendation.upper():
                self.attempt_mining(block_data, recommendation)
            else:
                log_event(f"[MINING] Skipping block based on recommendation: {recommendation}")
                
        except Exception as e:
            log_event(f"[ERROR] Block processing failed: {e}", "ERROR")
    
    def prepare_model_input(self, block_data: bytes) -> Dict[str, Any]:
        """Prepare structured input for model analysis"""
        # This would normally include mathematical validation results
        # For now, we'll create a basic structure
        return {
            "block_size": len(block_data),
            "timestamp": int(time.time()),
            "level": self.config['recursion_level'],
            "math_equation": self.config['math_equation'],
            "block_hash": block_data[:32].hex() if len(block_data) >= 32 else "unknown"
        }
    
    def get_mining_recommendation(self, validation_data: Dict[str, Any]) -> str:
        """Get mining recommendation using enhanced model interface"""
        self.stats['model_calls'] += 1
        
        try:
            # Create structured prompt
            prompt = f"""Analyze this Bitcoin mining validation data:
Level: {validation_data.get('level', 16000)}
Math Equation: {validation_data.get('math_equation', 'unknown')}
Block Size: {validation_data.get('block_size', 0)} bytes
Block Hash: {validation_data.get('block_hash', 'unknown')}

Provide mining recommendation: PROCEED, HOLD, or RETRY with brief reasoning."""

            model_input = ModelInput(
                prompt=prompt,
                context=validation_data,
                timeout=30,
                retry_attempts=3
            )
            
            result = self.model_interface.query_model_structured(model_input)
            
            if result.response_type == ModelResponseType.SUCCESS:
                self.stats['successful_calls'] += 1
                log_event(f"[MODEL] Success in {result.processing_time:.2f}s (retries: {result.retry_count})")
                return result.content or "HOLD"
            else:
                self.stats['failed_calls'] += 1
                log_event(f"[MODEL] Failed: {result.error_message}", "WARNING")
                
                # Fallback to mathematical decision
                return get_ai_recommendation_with_fallback(
                    validation_data, 
                    validation_data.get('block_hash', ''),
                    enable_ai=False
                )
                
        except Exception as e:
            self.stats['failed_calls'] += 1
            log_event(f"[MODEL] Exception: {e}", "ERROR")
            return "HOLD"
    
    def attempt_mining(self, block_data: bytes, recommendation: str) -> bool:
        """Attempt mining operation based on recommendation"""
        self.stats['mining_attempts'] += 1
        
        try:
            log_event(f"[MINING] Attempting mining based on: {recommendation[:50]}...")
            
            # This is where actual mining would occur
            # For now, we'll simulate the mining process
            time.sleep(0.1)  # Simulate mining work
            
            log_event("[MINING] Mining attempt completed")
            return True
            
        except Exception as e:
            log_event(f"[MINING] Failed: {e}", "ERROR")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mining statistics"""
        success_rate = 0.0
        if self.stats['model_calls'] > 0:
            success_rate = (self.stats['successful_calls'] / self.stats['model_calls']) * 100
            
        return {
            **self.stats,
            'model_success_rate': f"{success_rate:.1f}%",
            'health_check': self.model_interface.health_check()
        }
    
    def start(self) -> None:
        """Start the mining core system"""
        log_event("[INIT] SignalCore Bitcoin Mining Core Starting")
        
        self.running = True
        
        # Validate model interface
        if not self.model_interface.health_check():
            log_event("[MODEL] Health check failed, continuing with fallback mode", "WARNING")
        
        # Start ZMQ listener
        zmq_started = self.start_zmq_listener()
        if not zmq_started:
            log_event("[ZMQ] Using fallback polling mode", "WARNING")
        
        log_event("[INIT] Mining core started successfully")
        
        # Main loop
        try:
            while self.running:
                time.sleep(1)  # Main loop sleep
                
        except KeyboardInterrupt:
            log_event("[SHUTDOWN] Shutdown requested")
            self.stop()
    
    def stop(self) -> None:
        """Stop the mining core system"""
        self.running = False
        log_event("[SHUTDOWN] Mining core stopped")
        
        # Print final stats
        stats = self.get_stats()
        log_event(f"[STATS] Final statistics: {stats}")


def main():
    """Main entry point"""
    mining_core = BitcoinMiningCore()
    mining_core.start()


if __name__ == '__main__':
    main()
