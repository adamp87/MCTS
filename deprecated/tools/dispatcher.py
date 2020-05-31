from os import path
import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run(exePath, workDir):
    # run each process in parallel
    #exe params: cheat 0 writeTree 0 seed time+pid p2 1 workDir .
    result = subprocess.run([exePath, "workDir", workDir, "p0", "1", "p1", "1", "p2", "1", "p3", "2000"], capture_output=True)
    result = result.stdout
    return result

def pointDistributionTest(workDir, returns):
    # test to generate point histogram 
    pointsHist = np.zeros((28, 4), dtype=int)
    if path.isfile(workDir + 'points.npy') == True:
        pointsHist = np.load(workDir + 'points.npy')
    for ret in returns:
        result = ret.result()
        lines = result.splitlines()

        # parse points for players
        p1 = int(lines[-4].decode().split(' ')[1])
        p2 = int(lines[-3].decode().split(' ')[1])
        p3 = int(lines[-2].decode().split(' ')[1])
        p4 = int(lines[-1].decode().split(' ')[1])

        # shift points because of moon shooting
        p1 += 1
        p2 += 1
        p3 += 1
        p4 += 1
        if p1 == 27:
            p1 = 0
            p2 = 27
            p3 = 27
            p4 = 27
        elif p2 == 27:
            p2 = 0
            p1 = 27
            p3 = 27
            p4 = 27
        elif p3 == 27:
            p3 = 0
            p2 = 27
            p1 = 27
            p4 = 27
        elif p4 == 27:
            p4 = 0
            p2 = 27
            p3 = 27
            p1 = 27

        # increment histogram
        pointsHist[p1, 0] += 1
        pointsHist[p2, 1] += 1
        pointsHist[p3, 2] += 1
        pointsHist[p4, 3] += 1    
    np.save(workDir + 'points', pointsHist)

if __name__ == "__main__":
    exePath = '../build/release/Hearts'
    workDir = '../build/release/results/'

    workers = 4
    dispatchPerLoop = 4
    counter = 0;
    while True: # endless execution
        # execute gameagent in parallel
        returns = []
        executor = ThreadPoolExecutor(max_workers=workers)
        for i in range(dispatchPerLoop*workers):
            res = executor.submit(run, exePath, workDir)
            returns.append(res)
        executor.shutdown()

        # collect results and store
        pointDistributionTest(workDir, returns)

        counter += dispatchPerLoop*workers
        print ("Executed instances: " + str(counter))
        # program must be killed externally


