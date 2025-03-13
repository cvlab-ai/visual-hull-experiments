#!.test-env/bin/python3

import numpy as np
import matplotlib.pyplot as plt



def test(do_random = False):
    rng = np.random.default_rng()
    
    projections = []
    projections.append(xinf)
    rec = reconstruction(projections)
    return 0


if __name__ == "__main__":

    icp_mse_random = test(True)
    print(f"metric of random points {icp_mse_random}")

    mses = []
    sum_mse = 0
    for i in range(NUM_TESTS):
        icp_mse = test()
        print(f"{i}-th test MSE {icp_mse}")
        mses.append(icp_mse)

    avg_mse = np.average(mses)
    stdev_mse = np.std(mses)
    print(f"average MSE {avg_mse}")
    print(f"stdev of MSE {stdev_mse}")
