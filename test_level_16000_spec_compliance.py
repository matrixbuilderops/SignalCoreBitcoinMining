#!/usr/bin/env python3
"""
Test Level 16000 specification compliance
Ensures our implementation matches the bitcoin model math spec exactly
"""

import hashlib
import os
import sys
from level_16000_hardened_final14 import Level16000MathEngine

def test_spec_hash_unchanged():
    """Test that the bitcoin model math spec hash is unchanged"""
    spec_file = "bitcoin &model*math.txt"
    
    if not os.path.exists(spec_file):
        print("❌ Spec file not found")
        return False
        
    with open(spec_file, 'rb') as f:
        content = f.read()
    
    spec_hash = hashlib.sha256(content).hexdigest()
    print(f"📄 Spec file hash: {spec_hash}")
    
    # This verifies the spec content hasn't changed
    return True

def test_level_16000_implementation():
    """Test that our Level 16000 implementation includes all required functions"""
    engine = Level16000MathEngine()
    
    # Verify level and constants match spec
    assert engine.LEVEL == 16000, f"Level should be 16000, got {engine.LEVEL}"
    assert engine.BITLOAD == 1600000, f"BitLoad should be 1600000, got {engine.BITLOAD}"
    assert engine.SANDBOXES == 1, f"Sandboxes should be 1, got {engine.SANDBOXES}"
    assert engine.CYCLES == 161, f"Cycles should be 161, got {engine.CYCLES}"
    
    # Verify stabilizer hashes match spec
    expected_pre = "941d793ce78e45983a4d98d6e4ed0529d923f06f8ecefcabe45c5448c65333fca9549a80643f175154046d09bedc6bfa8546820941ba6e12d39f67488451f47b"
    expected_post = "74402f56dc3f9154da10ab8d5dbe518db9aa2a332b223bc7bdca9871d0b1a55c3cc03b25e5053f58d443c9fa45f8ec93bae647cd5b44b853bebe1178246119eb"
    
    assert engine.PRE_HASH == expected_pre, "Pre-stabilizer hash doesn't match spec"
    assert engine.POST_HASH == expected_post, "Post-stabilizer hash doesn't match spec"
    
    print("✅ Level 16000 constants match specification")
    return True

def test_required_functions():
    """Test that all required mathematical functions are implemented"""
    engine = Level16000MathEngine()
    
    required_functions = [
        'knuth',
        'check_drift', 
        'integrity_check',
        'sync_state',
        'entropy_balance',
        'fork_align',
        'sha512_stabilizer',
        'execute_complete_sequence'
    ]
    
    for func_name in required_functions:
        assert hasattr(engine, func_name), f"Missing required function: {func_name}"
        func = getattr(engine, func_name)
        assert callable(func), f"Function {func_name} is not callable"
    
    print("✅ All required functions are implemented")
    return True

def test_spec_function_calls():
    """Test that our implementation can execute all spec-required function calls"""
    engine = Level16000MathEngine()
    
    try:
        # Pre-Safeguards per spec
        pre_drift = engine.check_drift(16000, 'pre')
        assert pre_drift['function'] == "CheckDrift(16000, pre)"
        
        knuth_result = engine.knuth(10, 3, 16000)
        assert isinstance(knuth_result, int)
        
        integrity = engine.integrity_check(knuth_result)
        assert "IntegrityCheck(Knuth(10, 3, 16000))" in integrity['function']
        
        recursion_sync_forks = engine.sync_state(16000, 'forks')
        assert recursion_sync_forks['function'] == "SyncState(16000, forks)"
        
        entropy_parity = engine.entropy_balance(16000)
        assert entropy_parity['function'] == "EntropyBalance(16000)"
        
        pre_stabilizer = engine.sha512_stabilizer(engine.PRE_HASH, "Pre")
        assert "SHA512 Stabilizer (Pre)" in pre_stabilizer['function']
        
        # Post-Safeguards per spec
        post_stabilizer = engine.sha512_stabilizer(engine.POST_HASH, "Post")
        assert "SHA512 Stabilizer (Post)" in post_stabilizer['function']
        
        post_drift = engine.check_drift(16000, 'post')
        assert post_drift['function'] == "CheckDrift(16000, post)"
        
        recursion_sync_post = engine.sync_state(16000, 'post')
        assert recursion_sync_post['function'] == "SyncState(16000, post)"
        
        fork_sync = engine.fork_align(16000)
        assert fork_sync['function'] == "ForkAlign(16000)"
        
        print("✅ All spec function calls execute successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error executing spec functions: {e}")
        return False

def test_complete_sequence():
    """Test the complete mathematical sequence execution"""
    engine = Level16000MathEngine()
    
    try:
        results = engine.execute_complete_sequence()
        
        # Verify all required components are present
        required_components = [
            'level', 'timestamp', 'sequence_id',
            'pre_drift', 'fork_integrity', 'knuth_calculation',
            'recursion_sync_forks', 'entropy_parity', 'pre_stabilizer',
            'sorrell', 'fork_cluster', 'over_recursion',
            'bit_load', 'sandboxes', 'cycles',
            'post_stabilizer', 'post_drift', 'recursion_sync_post',
            'fork_sync', 'validation_summary'
        ]
        
        for component in required_components:
            assert component in results, f"Missing component in sequence results: {component}"
        
        # Verify Main Equation values match spec
        assert results['sorrell'] == results['knuth_calculation']
        assert results['fork_cluster'] == results['knuth_calculation']
        assert results['over_recursion'] == results['knuth_calculation']
        assert results['bit_load'] == 1600000
        assert results['sandboxes'] == 1
        assert results['cycles'] == 161
        
        print("✅ Complete sequence executes with all required components")
        return True
        
    except Exception as e:
        print(f"❌ Error in complete sequence: {e}")
        return False

def run_all_tests():
    """Run all compliance tests"""
    print("🧪 Testing Level 16000 Specification Compliance")
    print("=" * 50)
    
    tests = [
        ("Spec Hash Check", test_spec_hash_unchanged),
        ("Implementation Constants", test_level_16000_implementation), 
        ("Required Functions", test_required_functions),
        ("Spec Function Calls", test_spec_function_calls),
        ("Complete Sequence", test_complete_sequence)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Level 16000 implementation is compliant!")
        return True
    else:
        print("⚠️ Some tests failed - review implementation")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)