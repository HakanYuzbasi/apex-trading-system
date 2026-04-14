from quant_system.risk.mvo_sizer import MeanVarianceSizer

def test_sigmoid_allocations():
    sizer = MeanVarianceSizer(base_kelly_multiplier=0.5)
    
    probs_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("Testing Sigmoid MVO Allocations (Base Kelly 0.5):")
    for p in probs_to_test:
        allocs = sizer.allocate_strategies(p)
        b_weight = allocs["BreakoutPod"]
        k_weight = allocs["KalmanPairs"]
        print(f"Prob: {p:.2f} | Breakout: {b_weight:.3f} | Pairs: {k_weight:.3f} | Total: {b_weight + k_weight:.3f}")

if __name__ == "__main__":
    test_sigmoid_allocations()
