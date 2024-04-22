def scaler(x, min_x, max_x):
    # Define the scaling function based on your requirements
    # For now, it's a Min-Max Scaler
    return (x - min_x) / (max_x - min_x)

def asur(ba_b, asr_b, ba_a, asr_a, delta = -0.1):
    """
    Calculate ASuR based on the provided inputs.
    :param ba_b: BA before the defender's action, [0,1]
    :param asr_b: ASR before the defender's action, [0,1]
    :param ba_a: BA after the defender's action, [0,1]
    :param asr_a: ASR after the defender's action, [0,1]
    :param delta: Threshold δ
    :return: Calculated ASuR, [0,1]
    """
    alpha1, alpha2 = 0.95, 0.05  # Define alpha1 and alpha2 values based on requirements
    delta_ba = ba_a - ba_b
    delta_asr = asr_a - asr_b

    if delta_ba >= delta:
        asur_value = alpha1 * scaler(delta_asr, -asr_b, 1 - asr_b) + alpha2 * scaler(delta_ba, delta, 1 - ba_b) 
    else:
        alpha1, alpha2 = 0.5, 0.5
        asur_value = alpha1 * scaler(delta_asr, -asr_b, 1 - asr_b) + alpha2 * scaler(-delta_ba, -delta, ba_b)

    return asur_value

if __name__ == "__main__":
    print(asur(0.921, 0.719, 0.9099, 0.332)) # 0.3402
    print(asur(0.921, 0.719, 0.9122, 0.7143)) # 0.7041
    print(asur(0.8545, 0.9847, 0.2594, 0.9392)) # 79.77
    print(asur(0.8545, 0.9847, 0.8643, 0.4294)) # 0.4303