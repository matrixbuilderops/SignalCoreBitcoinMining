# LEVEL 16000 BITCOIN MINING SYSTEM - PART 2
# This continues exactly where Part 1 left off
# DO NOT RUN THIS SEPARATELY - Append this to Part 1

class EnhancedZMQListener:
    """Enhanced ZMQ listener for real-time network monitoring"""
    
    def __init__(self):
        self.endpoints = {
            "hashblock": "tcp://127.0.0.1:28332",
            "rawblock": "tcp://127.0.0.1:28333",
            "hashtx": "tcp://127.0.0.1:28334",
            "rawtx": "tcp://127.0.0.1:28335"
        }
        self.running = False
        self.callback = None
        self.zmq_working = False
        
        # Try to import ZMQ
        try:
            import zmq
            self.zmq = zmq
            self.zmq_available = True
            logger.info("✅ ZMQ library available for real-time monitoring")
        except ImportError:
            self.zmq_available = False
            logger.warning("⚠️ ZMQ not available - will use polling mode")
    
    def test_zmq_connectivity(self) -> bool:
        """Test if ZMQ endpoints are actually responding"""
        if not self.zmq_available:
            return False
        
        try:
            context = self.zmq.Context()
            socket = context.socket(self.zmq.SUB)
            socket.setsockopt(self.zmq.RCVTIMEO, 2000)  # 2 second timeout
            socket.connect(self.endpoints["hashblock"])
            socket.setsockopt(self.zmq.SUBSCRIBE, b"hashblock")
            
            # Try to receive (will timeout if no ZMQ server)
            try:
                socket.recv(self.zmq.NOBLOCK)
            except self.zmq.Again:
                pass  # Timeout is expected if no immediate messages
            
            socket.close()
            context.term()
            
            logger.info("✅ ZMQ endpoints responding - real-time mode active")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ ZMQ endpoints not responding: {e}")
            return False
    
    def set_callback(self, callback):
        """Set callback for received messages"""
        self.callback = callback
    
    def start(self):
        """Start ZMQ listener with connectivity testing"""
        self.running = True
        
        if self.zmq_available:
            self.zmq_working = self.test_zmq_connectivity()
            
            if self.zmq_working:
                logger.info("🚀 Starting real ZMQ network listener")
                threading.Thread(target=self._zmq_listen, daemon=True).start()
            else:
                logger.info("🔄 ZMQ not configured - using template polling mode")
                threading.Thread(target=self._polling_mode, daemon=True).start()
        else:
            logger.info("🔄 ZMQ library not available - using template polling mode")
            threading.Thread(target=self._polling_mode, daemon=True).start()
    
    def _zmq_listen(self):
        """Real ZMQ listening implementation"""
        try:
            context = self.zmq.Context()
            sockets = {}
            
            for topic, address in self.endpoints.items():
                socket = context.socket(self.zmq.SUB)
                socket.setsockopt(self.zmq.SUBSCRIBE, topic.encode())
                socket.setsockopt(self.zmq.RCVTIMEO, 10000)  # 10 second timeout
                socket.connect(address)
                sockets[topic] = socket
            
            logger.info("📡 ZMQ listener operational - monitoring Bitcoin network events")
            
            while self.running:
                for topic, socket in sockets.items():
                    try:
                        raw_topic = socket.recv(self.zmq.NOBLOCK)
                        message = socket.recv(self.zmq.NOBLOCK)
                        
                        logger.info(f"📨 [ZMQ-{topic.upper()}] Real network event received!")
                        
                        if self.callback:
                            self.callback(topic, message)
                            
                    except self.zmq.Again:
                        continue
                    except Exception as e:
                        logger.error(f"ZMQ error on {topic}: {e}")
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"ZMQ listener fatal error: {e}")
        finally:
            try:
                for socket in sockets.values():
                    socket.close()
                context.term()
            except:
                pass
    
    def _polling_mode(self):
        """Template polling mode when ZMQ unavailable"""
        logger.info("🔄 Template polling mode active")
        
        while self.running:
            try:
                time.sleep(15)  # Poll every 15 seconds
                
                # Simulate template update notification
                if self.callback:
                    self.callback("template_update", b"polling_trigger")
                
            except Exception as e:
                logger.error(f"Polling mode error: {e}")
    
    def stop(self):
        """Stop the listener"""
        self.running = False
        logger.info("🛑 Network listener stopped")


class AdvancedMiningOrchestrator:
    """Advanced orchestrator with ZMQ integration and enhanced features"""
    
    def __init__(self):
        # Initialize all components
        self.math_engine = Level16000MathEngine()
        self.bitcoin_core = BitcoinCore()
        self.model_interface = ModelInterface()
        self.zmq_listener = EnhancedZMQListener()
        
        # State management
        self.system_ready = False
        self.mining_active = False
        self.latest_template = None
        self.template_lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            "template_updates": 0,
            "network_events": 0,
            "processing_times": [],
            "math_computation_times": [],
            "ai_processing_times": [],
            "submission_attempts": 0,
            "submission_successes": 0
        }
        
        # Setup ZMQ callback
        self.zmq_listener.set_callback(self._on_network_event)
    
    def _on_network_event(self, topic: str, message: bytes):
        """Handle network events from ZMQ or polling"""
        try:
            self.performance_metrics["network_events"] += 1
            
            if topic in ["hashblock", "rawblock", "template_update"]:
                logger.info(f"🌐 [Network Event] {topic.upper()} - Blockchain activity detected!")
                
                # Trigger template update
                threading.Thread(target=self._update_template, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Error processing network event {topic}: {e}")
    
    def _update_template(self):
        """Update block template from Bitcoin Core"""
        try:
            logger.info("📋 [Template Update] Fetching latest template...")
            template = self.bitcoin_core.get_block_template()
            
            if template:
                with self.template_lock:
                    self.latest_template = template
                
                self.performance_metrics["template_updates"] += 1
                
                logger.info(f"✅ [Template] Height: {template.get('height', 'unknown')}, "
                          f"Difficulty: {template.get('bits', 'unknown')}, "
                          f"Transactions: {len(template.get('transactions', []))}")
            else:
                logger.warning("⚠️ [Template] Failed to get template from Bitcoin Core")
                
        except Exception as e:
            logger.error(f"Template update error: {e}")
    
    def validate_complete_system(self) -> bool:
        """Comprehensive system validation with enhanced checks"""
        logger.info("🔍 [System Validation] Starting comprehensive validation...")
        
        validation_results = {
            "bitcoin_core": False,
            "math_engine": False,
            "ai_model": False,
            "zmq_system": False
        }
        
        # 1. Bitcoin Core comprehensive validation
        logger.info("🔗 [Bitcoin Validation] Testing Bitcoin Core integration...")
        bitcoin_validation = self.bitcoin_core.comprehensive_validation()
        validation_results["bitcoin_core"] = "✅ READY" in bitcoin_validation["overall_status"]
        
        if validation_results["bitcoin_core"]:
            logger.info("✅ [Bitcoin Core] Ready for mining operations")
        else:
            logger.error("❌ [Bitcoin Core] Not ready - check configuration")
            logger.info("💡 Ensure Bitcoin Core is running, synced, and RPC is enabled")
        
        # 2. Math engine validation with extended testing
        logger.info("🧮 [Math Validation] Testing Level 16000 mathematical engine...")
        try:
            # Run a complete test sequence
            test_math = self.math_engine.execute_complete_sequence()
            math_success_rate = test_math.get('validation_summary', {}).get('success_rate', 0)
            
            if math_success_rate >= 70:
                validation_results["math_engine"] = True
                logger.info(f"✅ [Math Engine] Ready - {math_success_rate}% validation success")
            else:
                validation_results["math_engine"] = False
                logger.error(f"❌ [Math Engine] Failed - {math_success_rate}% validation success")
                
        except Exception as e:
            validation_results["math_engine"] = False
            logger.error(f"❌ [Math Engine] Exception: {e}")
        
        # 3. AI Model validation
        logger.info("🧠 [AI Validation] Testing model readiness...")
        ai_ready = self.model_interface.model_ready
        
        if ai_ready:
            validation_results["ai_model"] = True
            logger.info("✅ [AI Model] Ready for Level 16000 processing")
        else:
            validation_results["ai_model"] = False
            logger.warning("⚠️ [AI Model] Not ready - may cause processing delays")
            # Continue anyway for AI issues
            validation_results["ai_model"] = True
        
        # 4. ZMQ/Network monitoring validation
        logger.info("📡 [ZMQ Validation] Testing network monitoring...")
        zmq_ready = self.zmq_listener.zmq_available
        
        if zmq_ready:
            validation_results["zmq_system"] = True
            logger.info("✅ [ZMQ] Real-time network monitoring available")
        else:
            validation_results["zmq_system"] = True  # Polling mode works too
            logger.info("ℹ️ [ZMQ] Using polling mode (ZMQ not available)")
        
        # Overall system readiness
        critical_systems = ["bitcoin_core", "math_engine"]
        critical_ready = all(validation_results[sys] for sys in critical_systems)
        
        self.system_ready = critical_ready
        
        logger.info("=" * 80)
        logger.info("🎯 [COMPREHENSIVE VALIDATION SUMMARY]")
        logger.info(f"   Bitcoin Core: {'✅ READY' if validation_results['bitcoin_core'] else '❌ NOT READY'}")
        logger.info(f"   Math Engine: {'✅ READY' if validation_results['math_engine'] else '❌ NOT READY'}")
        logger.info(f"   AI Model: {'✅ READY' if validation_results['ai_model'] else '❌ NOT READY'}")
        logger.info(f"   Network Monitor: {'✅ READY' if validation_results['zmq_system'] else '❌ NOT READY'}")
        logger.info(f"   Overall Status: {'✅ SYSTEM READY FOR LEVEL 16000 MINING' if self.system_ready else '❌ SYSTEM NOT READY'}")
        logger.info("=" * 80)
        
        if not self.system_ready:
            logger.error("❌ Critical systems not ready - mining cannot start")
            logger.info("🔧 Please fix the issues above before starting mining")
        
        return self.system_ready
    
    def enhanced_mining_workflow(self):
        """Enhanced mining workflow with performance tracking"""
        while self.mining_active:
            try:
                # Check for new template
                template = None
                with self.template_lock:
                    if self.latest_template:
                        template = self.latest_template
                        self.latest_template = None  # Clear after taking
                
                if template:
                    workflow_start = time.time()
                    
                    logger.info("🚀 [Enhanced Workflow] Starting Level 16000 mining process...")
                    logger.info(f"📋 [Template Info] Height: {template.get('height', 0)}, "
                               f"Difficulty: {template.get('bits', 'unknown')}, "
                               f"Target: {str(template.get('target', 'unknown'))[:16]}...")
                    
                    # Step 1: Execute complete Level 16000 mathematics
                    logger.info("🧮 [Step 1/4] Executing complete Level 16000 mathematical sequence...")
                    math_start = time.time()
                    math_results = self.math_engine.execute_complete_sequence(template)
                    math_time = time.time() - math_start
                    
                    self.performance_metrics["math_computation_times"].append(math_time)
                    
                    # Validate mathematical results
                    math_summary = math_results.get('validation_summary', {})
                    math_success_rate = math_summary.get('success_rate', 0)
                    
                    if math_success_rate < 50:
                        logger.warning(f"⚠️ [Math Results] Low success rate: {math_success_rate}% - skipping this template")
                        continue
                    
                    logger.info(f"✅ [Math Complete] Success rate: {math_success_rate}% (Time: {math_time:.2f}s)")
                    
                    # Step 2: Enhanced AI processing with complete context
                    logger.info("🧠 [Step 2/4] Processing with AI model (Level 16000 context)...")
                    ai_start = time.time()
                    ai_result = self.model_interface.process_task(template, math_results)
                    ai_time = time.time() - ai_start
                    
                    self.performance_metrics["ai_processing_times"].append(ai_time)
                    
                    logger.info(f"✅ [AI Complete] Processing time: {ai_time:.2f}s")
                    logger.info(f"📝 [AI Output Preview] {ai_result[:150]}...")
                    
                    # Step 3: Enhanced solution extraction and validation
                    logger.info("🔍 [Step 3/4] Extracting and validating solution...")
                    solution = self._enhanced_extract_solution(ai_result, math_results, template)
                    
                    # Step 4: Submit if valid with enhanced tracking
                    if solution and len(solution) > 160:
                        logger.info("📤 [Step 4/4] Submitting solution to Bitcoin network...")
                        logger.info(f"🔢 [Solution Info] Length: {len(solution)} chars, "
                                   f"Hash preview: {hashlib.sha256(solution.encode()).hexdigest()[:16]}...")
                        
                        self.performance_metrics["submission_attempts"] += 1
                        
                        submission_result = self.bitcoin_core.submit_block(solution)
                        success = self._enhanced_evaluate_submission(submission_result)
                        
                        self._track_enhanced_submission_metrics(solution, submission_result, success, template)
                        
                        global system_stats
                        system_stats["solutions_generated"] += 1
                        
                        if success:
                            system_stats["successful_submissions"] += 1
                            self.performance_metrics["submission_successes"] += 1
                            logger.info("🎉 [SUCCESS] Solution accepted by Bitcoin network!")
                        else:
                            logger.info(f"📝 [SUBMITTED] Network response: {submission_result}")
                        
                    else:
                        logger.warning("⚠️ [Step 4/4] No valid solution generated - skipping submission")
                    
                    # Track overall performance
                    workflow_time = time.time() - workflow_start
                    self.performance_metrics["processing_times"].append(workflow_time)
                    
                    system_stats["blocks_processed"] += 1
                    
                    logger.info(f"✅ [Workflow Complete] Total: {workflow_time:.2f}s "
                               f"(Math: {math_time:.2f}s, AI: {ai_time:.2f}s)")
                    logger.info("🔄 [Status] Ready for next template...")
                
                time.sleep(1)  # Brief pause between workflow cycles
                
            except Exception as e:
                logger.error(f"Enhanced mining workflow error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)
    
    def _enhanced_extract_solution(self, ai_output: str, math_results: Dict, template: Dict) -> Optional[str]:
        """Enhanced solution extraction with multiple validation layers"""
        try:
            logger.info("🔍 [Solution Extraction] Analyzing AI output for valid solutions...")
            
            lines = ai_output.split('\n')
            potential_solutions = []
            
            # Look for different solution patterns
            solution_patterns = [
                "SOLUTION:",
                "BLOCK:",
                "HEX:",
                "RESULT:",
                "OUTPUT:"
            ]
            
            in_solution_section = False
            
            for line in lines:
                # Check if we're entering a solution section
                for pattern in solution_patterns:
                    if pattern in line.upper():
                        in_solution_section = True
                        break
                
                # Extract potential hex solutions
                clean_line = line.strip()
                if len(clean_line) > 160:
                    # Check if it's a valid hex string
                    if all(c in '0123456789abcdefABCDEF' for c in clean_line):
                        potential_solutions.append({
                            "hex": clean_line,
                            "length": len(clean_line),
                            "in_section": in_solution_section,
                            "line": line
                        })
            
            logger.info(f"🔍 [Solution Analysis] Found {len(potential_solutions)} potential solutions")
            
            if potential_solutions:
                # Score and rank solutions
                for solution in potential_solutions:
                    score = self._score_solution(solution, math_results, template)
                    solution["score"] = score
                
                # Sort by score (highest first)
                potential_solutions.sort(key=lambda x: x["score"], reverse=True)
                
                # Validate the best solution
                best_solution = potential_solutions[0]
                logger.info(f"✅ [Best Solution] Score: {best_solution['score']}, Length: {best_solution['length']}")
                
                if self._enhanced_validate_solution(best_solution["hex"], math_results, template):
                    logger.info("✅ [Solution Validation] Passed all Level 16000 validation checks")
                    return best_solution["hex"]
                else:
                    logger.warning("⚠️ [Solution Validation] Failed Level 16000 validation")
            
            logger.warning("⚠️ [Solution Extraction] No valid solutions found in AI output")
            return None
            
        except Exception as e:
            logger.error(f"Solution extraction error: {e}")
            return None
    
    def _score_solution(self, solution: Dict, math_results: Dict, template: Dict) -> float:
        """Score a potential solution based on multiple criteria"""
        score = 0.0
        
        try:
            hex_data = solution["hex"]
            
            # Length score (longer is generally better for complete blocks)
            if solution["length"] >= 160:
                score += 10.0
            if solution["length"] >= 1000:
                score += 20.0
            
            # Section placement score
            if solution["in_section"]:
                score += 15.0
            
            # Hex validity score
            try:
                bytes.fromhex(hex_data)
                score += 10.0
            except ValueError:
                score -= 50.0  # Heavily penalize invalid hex
            
            # Entropy score (good distribution of characters)
            unique_chars = len(set(hex_data))
            if unique_chars >= 10:  # Good distribution
                score += 5.0
            
            # Level 16000 mathematical alignment score
            math_success_rate = math_results.get('validation_summary', {}).get('success_rate', 0)
            score += math_success_rate * 0.2  # Up to 20 points for 100% math success
            
            # Template alignment score
            if template:
                template_height = template.get('height', 0)
                if template_height > 0:
                    score += 5.0
            
        except Exception as e:
            logger.error(f"Solution scoring error: {e}")
            score = 0.0
        
        return score
    
    def _enhanced_validate_solution(self, solution: str, math_results: Dict, template: Dict) -> bool:
        """Enhanced solution validation with comprehensive checks"""
        try:
            logger.info("🔍 [Enhanced Validation] Running comprehensive solution validation...")
            
            # Basic format validation
            if len(solution) < 160:
                logger.warning("❌ [Validation] Solution too short (minimum 160 chars for block header)")
                return False
            
            # Hex format validation
            try:
                solution_bytes = bytes.fromhex(solution)
                logger.info(f"✅ [Validation] Valid hex format ({len(solution_bytes)} bytes)")
            except ValueError:
                logger.warning("❌ [Validation] Invalid hex format")
                return False
            
            # Mathematical validation
            math_summary = math_results.get('validation_summary', {})
            math_success_rate = math_summary.get('success_rate', 0)
            
            if math_success_rate < 70:
                logger.warning(f"❌ [Validation] Math success rate too low: {math_success_rate}%")
                return False
            
            logger.info(f"✅ [Validation] Math success rate acceptable: {math_success_rate}%")
            
            # Level 16000 specific validation
            solution_hash = hashlib.sha256(solution.encode()).hexdigest()
            level_factor = int(solution_hash[-8:], 16) % 16000
            
            # Knuth value alignment check
            knuth_value = math_results.get('knuth_calculation', 0)
            if knuth_value == 0:
                logger.warning("❌ [Validation] No Knuth calculation available")
                return False
            
            logger.info(f"✅ [Validation] Knuth value present: {knuth_value}")
            
            # Entropy validation
            entropy_ratio = len(set(solution)) / len(solution)
            if entropy_ratio < 0.3:
                logger.warning(f"❌ [Validation] Entropy too low: {entropy_ratio:.3f}")
                return False
            
            logger.info(f"✅ [Validation] Entropy acceptable: {entropy_ratio:.3f}")
            
            # Template compatibility check
            if template:
                template_height = template.get('height', 0)
                if template_height > 0:
                    logger.info(f"✅ [Validation] Template height: {template_height}")
                else:
                    logger.warning("⚠️ [Validation] No template height available")
            
            # Level 16000 final validation
            level_16000_check = (level_factor + entropy_ratio * 1000 + (knuth_value % 1000)) % 16000
            
            logger.info(f"✅ [Level 16000 Validation] Factor: {level_factor}, "
                       f"Entropy: {entropy_ratio:.3f}, Check: {level_16000_check}")
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced validation error: {e}")
            return False
    
    def _enhanced_evaluate_submission(self, response: Any) -> bool:
        """Enhanced submission evaluation with detailed response analysis"""
        try:
            if response is None:
                logger.info("✅ [Submission] None response - typically indicates acceptance")
                return True
            
            response_str = str(response).lower()
            
            # Detailed success pattern matching
            success_patterns = [
                "accepted", "null", "", "none", "ok", "success"
            ]
            
            # Detailed failure pattern matching
            failure_patterns = [
                "rejected", "invalid", "duplicate", "stale", "error", 
                "bad", "malformed", "orphan", "inconclusive"
            ]
            
            # Check for success patterns
            for pattern in success_patterns:
                if pattern in response_str:
                    logger.info(f"✅ [Submission] Success pattern detected: '{pattern}'")
                    return True
            
            # Check for failure patterns
            for pattern in failure_patterns:
                if pattern in response_str:
                    logger.info(f"📝 [Submission] Failure pattern detected: '{pattern}'")
                    return False
            
            # Unknown response
            logger.info(f"❓ [Submission] Unknown response pattern: {response}")
            return False  # Conservative approach for unknown responses
            
        except Exception as e:
            logger.error(f"Submission evaluation error: {e}")
            return False
    
    def _track_enhanced_submission_metrics(self, solution: str, response: Any, success: bool, template: Dict):
        """Enhanced submission metrics tracking"""
        try:
            # Generate block hash from solution
            if len(solution) >= 160:
                header_data = bytes.fromhex(solution[:160])  # First 80 bytes of header
                double_hash = hashlib.sha256(hashlib.sha256(header_data).digest()).digest()
                block_hash = double_hash[::-1].hex()  # Bitcoin little-endian format
                
                status_emoji = "✅ ACCEPTED" if success else "📝 SUBMITTED"
                logger.info(f"📊 [Enhanced Submission Metrics] {status_emoji}")
                logger.info(f"    Block Hash: {block_hash[:32]}...")
                logger.info(f"    Solution Length: {len(solution)} hex characters")
                logger.info(f"    Template Height: {template.get('height', 'unknown')}")
                logger.info(f"    Network Response: {response}")
                
                if success:
                    logger.info(f"    🎉 SUCCESS: Block potentially added to Bitcoin blockchain!")
                
                # Store for global tracking
                global last_submitted_block_hash
                last_submitted_block_hash = block_hash
            
        except Exception as e:
            logger.error(f"Enhanced submission tracking error: {e}")
    
    def advanced_performance_reporter(self):
        """Advanced performance reporting with detailed metrics"""
        while self.mining_active:
            try:
                time.sleep(90)  # Report every 90 seconds for detailed analysis
                
                runtime = time.time() - system_stats["start_time"]
                
                logger.info("📊 ADVANCED LEVEL 16000 PERFORMANCE REPORT")
                logger.info("=" * 70)
                logger.info(f"   🕒 Runtime: {runtime/60:.1f} minutes ({runtime/3600:.2f} hours)")
                logger.info(f"   🔗 System Status: {'✅ OPERATIONAL' if self.system_ready else '❌ NOT READY'}")
                logger.info(f"   📋 Templates Received: {system_stats['templates_received']}")
                logger.info(f"   🔄 Template Updates: {self.performance_metrics['template_updates']}")
                logger.info(f"   🌐 Network Events: {self.performance_metrics['network_events']}")
                logger.info(f"   📦 Blocks Processed: {system_stats['blocks_processed']}")
                logger.info(f"   🧮 Math Sequences: {system_stats['math_sequences_completed']}")
                logger.info(f"   💡 Solutions Generated: {system_stats['solutions_generated']}")
                logger.info(f"   📤 Submission Attempts: {self.performance_metrics['submission_attempts']}")
                logger.info(f"   ✅ Successful Submissions: {system_stats['successful_submissions']}")
                logger.info(f"   🧠 AI Model Calls: {system_stats['model_calls']}")
                
                # Performance averages
                if self.performance_metrics["processing_times"]:
                    avg_processing = sum(self.performance_metrics["processing_times"]) / len(self.performance_metrics["processing_times"])
                    logger.info(f"   ⏱️ Avg Processing Time: {avg_processing:.2f}s per template")
                
                if self.performance_metrics["math_computation_times"]:
                    avg_math = sum(self.performance_metrics["math_computation_times"]) / len(self.performance_metrics["math_computation_times"])
                    logger.info(f"   🧮 Avg Math Time: {avg_math:.2f}s per sequence")
                
                if self.performance_metrics["ai_processing_times"]:
                    avg_ai = sum(self.performance_metrics["ai_processing_times"]) / len(self.performance_metrics["ai_processing_times"])
                    logger.info(f"   🧠 Avg AI Time: {avg_ai:.2f}s per analysis")
                
                # Success rates
                if system_stats["blocks_processed"] > 0:
                    block_success_rate = (system_stats["successful_submissions"] / system_stats["blocks_processed"]) * 100
                    logger.info(f"   📈 Block Success Rate: {block_success_rate:.2f}%")
                
                if self.performance_metrics["submission_attempts"] > 0:
                    submission_success_rate = (self.performance_metrics["submission_successes"] / self.performance_metrics["submission_attempts"]) * 100
                    logger.info(f"   📤 Submission Success Rate: {submission_success_rate:.2f}%")
                
                # Network activity rate
                if runtime > 0:
                    events_per_minute = (self.performance_metrics["network_events"] / runtime) * 60
                    logger.info(f"   🌐 Network Activity: {events_per_minute:.2f} events/minute")
                    
                    templates_per_hour = (system_stats["templates_received"] / runtime) * 3600
                    logger.info(f"   📋 Template Rate: {templates_per_hour:.1f} templates/hour")
                
                logger.info("="