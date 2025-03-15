import numpy as np

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400

class FlappyBirdAI:
    ## State function to calculate where the Bird currently is to submit it to the network
    def get_state(self, bird, pipes):
        if not pipes:
            return np.array([
                bird.y / SCREEN_HEIGHT, 
                bird.velocity / 10,
                1.0,
                0.5
            ])
        
        next_pipe = None
        #  for pipe in pipes:
        #     if pipe.x>bird.x:
        #         next_pipe = pipe
        #         break
        
        for pipe in pipes:
            if pipe.x+50> bird.x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = pipes[0]
        # print(bird.y/SCREEN_HEIGHT)
        return np.array([
            bird.y / SCREEN_HEIGHT,
            bird.velocity / 10,
            (next_pipe.x - bird.x) / SCREEN_WIDTH,
            next_pipe.height / SCREEN_HEIGHT
        ])
