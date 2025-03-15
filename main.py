import pygame
import random
import sys
import numpy as np
import time
import yaml
import os
import torch
from flappybirdAI import FlappyBirdAI
from QNetwork import GA_Network, mutate_network, crossover_networks

## game dependent stuff
class Bird:
    def __init__(self):
        self.x = SCREEN_WIDTH // 3
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.size = 20
        ## not zero since 0 is black xd
        r, g, b = [random.randint(50, 255) for _ in range(3)]
        self.color = (r, g, b)

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, int(self.y)), self.size)



class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(
            PIPE_MIN_HEIGHT, 
            min(PIPE_MAX_HEIGHT, SCREEN_HEIGHT - PIPE_GAP - 50)
        )
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, screen):
        pygame.draw.rect(
            screen, (0, 255, 0),
            (self.x, 0, 50, self.height)
        )
        pygame.draw.rect(
            screen, (0, 255, 0),
            (self.x, self.height + PIPE_GAP, 50, SCREEN_HEIGHT - self.height - PIPE_GAP)
        )


## Read config -> partially done with chatGpt and https://www.geeksforgeeks.org/reading-and-writing-yaml-file-in-python/
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

population_size = config.get("population_size", 200)
mutation_prob = config.get("mutation_probability", 0.05)
mutation_std = config.get("mutation_standard_deviation", 0.1)
elitism = config.get("elitism", True)

stop_cond = config.get("stop_condition", {})
max_generations = stop_cond.get("generations", float("inf"))
max_score = stop_cond.get("score", float("inf"))
max_time = stop_cond.get("time", float("inf"))

enable_acceleration = config.get("enable_acceleration", False)
save_best = config.get("save_best", True)
load_previous_best = config.get("load_previous_best", True)

hardstuck_gen = config.get("hardstuck_gen", 10)

level_config = config.get("level", {})
## constants
PIPE_GAP = level_config.get("pipe_gap", 150)
PIPE_MIN_HEIGHT = level_config.get("pipe_min_height", 50)
PIPE_MAX_HEIGHT = level_config.get("pipe_max_height", 300)
PIPE_SPEED = level_config.get("pipe_speed", 4)


## Pygame Setup
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GRAVITY = 0.5
FLAP_STRENGTH = -8

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("FlAIpy Bird - GA")
clock = pygame.time.Clock()
FONT = pygame.font.Font(None, 32)

### Bascially check if the bird pos is in the pipes. pos and returns a boolena
def check_collision(bird, pipes):
    for pipe in pipes:
        if (bird.x + bird.size > pipe.x and
            bird.x - bird.size < pipe.x + 50 and
            (bird.y - bird.size < pipe.height or
             bird.y + bird.size > pipe.height + PIPE_GAP)):
            return True
    if bird.y < 0 or bird.y > SCREEN_HEIGHT:
        return True
    return False

## reward funciton
def calculate_reward(bird, pipes, done):
    if done:
        return -1.0
    reward = 0.1
    if pipes:
        next_pipe = pipes[0]
        if next_pipe.x + 50 < bird.x and not next_pipe.passed:
            reward += 1.0
            next_pipe.passed = True

        gap_center = next_pipe.height + PIPE_GAP / 2
        alignment_penalty = -abs(bird.y - gap_center) / SCREEN_HEIGHT
        reward += alignment_penalty * 0.5
    return reward


def save_best_model(params, filename="best_model.pth"):
    torch.save(params, filename)

def load_best_model(filename="best_model.pth"):
    if not os.path.exists(filename):
        return None
    return torch.load(filename)

## CVreates a set of Populations
def create_population():
    birds = [Bird() for _ in range(population_size)]
    ai_agents = [FlappyBirdAI() for _ in range(population_size)]
    ##create a network with 4 input layers 128 hidden and 2 output -> do flap or not!
    networks = [GA_Network(4, 128, 2) for _ in range(population_size)]
    return birds, ai_agents, networks

# Runs one  genreratuin for each bird to make a fittness score network outputs action=argmax
# returns an array of scores for the population
def evaluate_generation(generation, birds, ai_agents, networks):

    pipes = [Pipe()]
    scores = np.zeros(len(birds))

    # Initialize "states" from AI
    states = [ai.get_state(bird, pipes).reshape(1, -1)
              for bird, ai in zip(birds, ai_agents)]

    done = [False] * len(birds)
    ## main loop while a bird is still allive
    while not all(done):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if pipes and pipes[-1].x <= SCREEN_WIDTH - 200:
            pipes.append(Pipe())

        for i, bird in enumerate(birds):
            if done[i]:
                continue

            # no epsilon just an  argmax
            q_values = networks[i].forward_prop(states[i])
            action = np.argmax(q_values)  # action s described in Readme 0 => do nothing, 1 => flap

            if action == 1:
                bird.flap()
            bird.update()

            if check_collision(bird, pipes):
                done[i] = True
            else:
                reward = calculate_reward(bird, pipes, done[i])
                scores[i] += reward
                ## update agents
                states[i] = ai_agents[i].get_state(bird, pipes).reshape(1, -1)

        for pipe in pipes:
            pipe.update()

        # background
        screen.fill((0, 0, 0))
        for pipe in pipes:
            pipe.draw(screen)
        for i, bird in enumerate(birds):
            if not done[i]:
                bird.draw(screen)

        info_texts = [
            f"Generation: {generation + 1}",
            f"Alive: {len(birds) - sum(done)}",
            f"Best Score (gen): {int(np.max(scores))}"
        ]

        for idx, text in enumerate(info_texts):
            surface = FONT.render(text, True, (255, 255, 255))
            screen.blit(surface, (10, 10 + idx * 20))

        pygame.display.flip()

        # If acceleration => no FPS limit
        if(enable_acceleration):
            clock.tick(0)
        else:
            clock.tick(60)
        

    return scores

## much help of chat gpt since i ran into soo much errors
def reproduce_population(networks, scores, elitism, mutation_prob, mutation_std):
    # selection + crossover + mutation = new populatio
    pop_size = len(networks)
    
    
    ## reason why it works better with 200 generation tested it with 20 but didn'T work at all a
    ## take the best 100 percent
    num_elites = max(1, pop_size // 10)
    idx_sorted = np.argsort(scores)[::-1]

    # Extract top networks
    elites = [networks[i] for i in idx_sorted[:num_elites]]

    # Create new population
    new_population = []
    
    ## eliten clonen und neu benutzen -> keine vielfallt bei zu vielen  eliten
    if elitism:
        for e in elites:
            cloned_net = GA_Network(4, 128, 2)
            cloned_net.set_params(e.get_params())
            new_population.append(cloned_net)

    ## restliche population mahcen
    while len(new_population) < pop_size:
        parent_indices = idx_sorted[: max(num_elites * 5, pop_size // 4)]
        p1_idx = random.choice(parent_indices)
        p2_idx = random.choice(parent_indices)

        child_net = GA_Network(4, 128, 2)
        
        # https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm)
        child_params = crossover_networks(networks[p1_idx], networks[p2_idx])
        child_net.set_params(child_params)
        
        mutate_network(child_net, mutation_prob, mutation_std)
        new_population.append(child_net)

    
    return new_population



def main():
    start_time = time.time()
    overall_highest_score = 0
    stuck_counter = 0
    generation = 0

    birds, ai_agents, networks = create_population()

    
    if load_previous_best:
        best_params = load_best_model("best_model.pth")
        if best_params is not None:
            for net in networks:
                net.set_params(best_params)
            print("DEBUG :> Loaded previous best model from best_model.pth")

    while generation < max_generations:
        if time.time() - start_time >= max_time:
            print("Reached max training time. Stopping.")
            break

        scores = evaluate_generation(generation, birds, ai_agents, networks)
        gen_best_score = np.max(scores)
        overall_highest_score = max(overall_highest_score, gen_best_score)

        
        if gen_best_score <= 0:
            stuck_counter += 1
        else:
            stuck_counter = 0
        # print(stuck_counter)

        ## not really reaching hecnec my reward function is always above 0
        if stuck_counter >= hardstuck_gen:
            print(f"population stuck for {hardstuck_gen} gens at 0 score Resetting population")
            birds, ai_agents, networks = create_population()
            stuck_counter = 0

        print(f"best score: {gen_best_score}")
        if gen_best_score >= max_score:
            print("Reached max score")
            break
        best_index = np.argmax(scores)
        best_params = networks[best_index].get_params()

        if save_best and gen_best_score == overall_highest_score:
            save_best_model(best_params, "best_model.pth")
            print("DEBUG :>  Saved new best model to best_model.pth")


        ## Spawn new generation
        networks = reproduce_population(networks, scores,elitism,
                                        mutation_prob,
                                        mutation_std)

        birds = [Bird() for _ in range(population_size)]
        ai_agents = [FlappyBirdAI() for _ in range(population_size)]

        generation += 1

    pygame.quit()


if __name__ == "__main__":
    main()
