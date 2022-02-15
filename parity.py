import numpy as np

def get3ParityDataset(batch=None, random_state=10):
    if batch is None:
        vecs = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], 
                [1,0,0], [1,0,1], [1,1,0], [1,1,1]]

        labs = [[0], [1], [1], [0], [1], [0], [0], [1]]
        
    else:
        rstate = np.random.RandomState(random_state)
        
        vecs = [rstate.randint(0,2,3) for _ in range(batch)]
        
        labs = [[0] if np.sum(x) % 2 == 0 else [1] for x in vecs]
        
    return np.array(vecs), np.array(labs)