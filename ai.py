import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import pygame
import matplotlib.pyplot as plt
import time

cycles_total = 300
breakthrough_cycle = 50

# Zmienne do ewaluacji czasu
total_eval_episodes = 6
max_eval_time = 10.0

# Marchewki i kije
class AntiCheatingWrapper(gym.Wrapper):
    def __init__(self, env, angle_threshold=0.12, max_tilted_steps=15, penalty=1.0, position_weight=0.6):
        super().__init__(env)
        # 0.08 radianów to około 4.5 stopnia
        self.angle_threshold = angle_threshold 
        self.max_tilted_steps = max_tilted_steps
        self.penalty = penalty
        self.position_weight = position_weight
        
        # Licznik, ile kolejnych kroków pole jest przechylone
        self.tilted_frames = 0

    def reset(self, **kwargs):
        self.tilted_frames = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # obs[0] to pozycja wózka (x), obs[2] to reprezentuje kąt nachylenia
        cart_position = obs[0]
        pole_angle = obs[2]
        
        # Sprawdzamy, czy kąt nachylenia przekracza nasz próg.
        if abs(pole_angle) > self.angle_threshold:
            self.tilted_frames += 1
        else:
            self.tilted_frames = 0
            
        # Odejmujemy punkty za zbyt długie przechylanie się
        if self.tilted_frames >= self.max_tilted_steps:
            reward -= self.penalty
            
        # Zakończenie odcinka po przewinieniu
        if self.tilted_frames >= self.max_tilted_steps * 3:
            terminated = True

        # Kara za oddalenie się od centrum toru
        # Wózek porusza się zwykle w przedziale od -2.4 do 2.4. Centrum to 0.0.
        # Obliczamy dystans od zera i mnożymy przez wagę kary.
        distance_from_center = abs(cart_position)
        position_penalty = distance_from_center * self.position_weight
        
        # Odejmujemy karę od standardowej nagrody (+1.0 za przetrwaną klatkę)
        reward -= position_penalty
            
        return obs, reward, terminated, truncated, info

class ExperimentCallback(BaseCallback):
    def __init__(self, exp_name="", start_cycles=0, target_cycles=breakthrough_cycle, total_cycles=cycles_total, verbose=0):
        super().__init__(verbose)
        self.generations = start_cycles
        self.exp_name = exp_name
        self.target_cycles = target_cycles
        self.total_cycles = total_cycles
        self.aborted = False
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 30, bold=True)
        self.last_drawn_gen = -1

    def _on_step(self):
        if self.locals.get("dones")[0]:
            self.generations += 1
        
        # Ograniczamy rysowanie tylko dla wizualizacji
        if self.generations != self.last_drawn_gen:
            self.last_drawn_gen = self.generations
            screen = pygame.display.get_surface()
            if screen:
                screen.fill((10, 10, 25))
                text1 = self.font.render(f"Experiment {self.exp_name}", True, (255, 100, 0))
                text2 = self.font.render(f"Cycles: {self.generations}/{self.total_cycles}", True, (255, 255, 50))
                screen.blit(text1, (20, 20))
                screen.blit(text2, (20, 60))
                pygame.display.update()
            
        pygame.event.pump()
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            self.aborted = True
            return False
            
        if self.generations >= self.target_cycles:
            return False
            
        return True

def run_experiment():
    pole_lengths = [0.2, 0.5, 1.0, 1.5]
    results = {}

    global_abort = False
    
    # Inicjalizacja stałego okna treningowego
    pygame.init()
    base_window_size = (600, 400)
    pygame.display.set_mode(base_window_size)
    pygame.display.set_caption("AI Quick Training")
    watch_font = pygame.font.SysFont("Arial", 30, bold=True)
    
    for length in pole_lengths:
        if global_abort: break

        # Środowisko treningowe BEZ renderowania
        train_env = gym.make("CartPole-v1", render_mode=None)
        
        # Zmiana fizycznych właściwości środowiska
        train_env.unwrapped.length = length
        train_env.unwrapped.polemass_length = train_env.unwrapped.masspole * length
        
        # Owinięcie środowiska naszym własnym algorytmem karania!
        wrapped_env = AntiCheatingWrapper(train_env, angle_threshold=0.12, max_tilted_steps=15, penalty=1.0)
        
        wrapped_env.reset()
        
        # Increase the size of the neural network (default is 64x64)
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        
        # PPO z kilkoma modyfikacjami hiperparametrów, które przyspieszają naukę kosztem stabilności:
        # n_steps=256 (default 2048): Szybsze aktualizacje modelu, ale mniej stabilne.
        # gamma=0.98 (default 0.99): Mniejszy współczynnik dyskontowania, co może przyspieszyć naukę krótkoterminowych strategii.
        model = PPO("MlpPolicy", wrapped_env, ent_coef=0.01, learning_rate=0.0005, n_steps=256, batch_size=64, gamma=0.98, policy_kwargs=policy_kwargs, verbose=0)
        exp_name = f"Pole Length: {length}"
        
        length_scores = []
        current_cycle = 0
        
        
        while current_cycle < cycles_total:
            # Upewnijmy się, że okno Pygame istnieje w fazie treningu (po ewaluacji mogło zniknąć)
            if not pygame.get_init():
                pygame.init()
            if pygame.display.get_surface() is None or pygame.display.get_surface().get_size() != base_window_size:
                pygame.display.set_mode(base_window_size)
                
            target_cycle = current_cycle + breakthrough_cycle
            callback = ExperimentCallback(exp_name=exp_name, start_cycles=current_cycle, target_cycles=target_cycle, total_cycles=cycles_total)
            
            model.learn(total_timesteps=200000, callback=callback, reset_num_timesteps=False)
            
            if callback.aborted:
                global_abort = True
                break
                
            current_cycle += breakthrough_cycle
            
            # Ewaluacja modelu w środowisku z renderowaniem, by zobaczyć postępy i zebrać dane do wykresu
            eval_env = gym.make("CartPole-v1", render_mode="human")
            eval_env.unwrapped.length = length
            eval_env.unwrapped.polemass_length = eval_env.unwrapped.masspole * length
            eval_wrapped = AntiCheatingWrapper(eval_env, angle_threshold=0.12, max_tilted_steps=15, penalty=1.0)
            
            if not pygame.get_init():
                pygame.init()
            if not pygame.font.get_init():
                pygame.font.init()
            watch_font = pygame.font.SysFont("Arial", 30, bold=True)
            
            obs, info = eval_wrapped.reset()
            
            total_score = 0
            episodes_tested = 0
            
            # Rozpocznij testowanie i mierzenie czasu przetrwania
            episode_start_time = time.time()
           
            
            while episodes_tested < total_eval_episodes:
                # W trybie single-env model.predict też zwraca poprawne akcje
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_wrapped.step(action)
                done = terminated or truncated
                
                # We track exactly 1 frame per step, instead of accumulating the polluted reward
                total_score += 1
                
                time_elapsed = time.time() - episode_start_time
                if done or time_elapsed > max_eval_time:
                    episodes_tested += 1
                    episode_start_time = time.time() # Resetuj czas dla następnego odcinka
                    if episodes_tested < total_eval_episodes:
                        obs, info = eval_wrapped.reset()
                    
                screen = pygame.display.get_surface()
                if screen:
                    text1 = watch_font.render(f"Testing {exp_name} @ Phase {current_cycle}/{cycles_total}", True, (0, 200, 255))
                    text2 = watch_font.render(f"Eval Episode {episodes_tested+1}/{total_eval_episodes} (Max 5s)", True, (50, 200, 50))
                    text3 = watch_font.render(f"Time: {time_elapsed:.1f}s", True, (200, 200, 200))
                    screen.blit(text1, (20, 20))
                    screen.blit(text2, (20, 60))
                    screen.blit(text3, (20, 100))
                    pygame.display.update()
                    
                pygame.event.pump()
                if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                    global_abort = True
                    break
                    
            eval_wrapped.close() # Zamykamy środowisko wizualne, by powrócić do super-szybkiego treningu
            
            if global_abort:
                break
                
            avg_score = total_score / total_eval_episodes
            length_scores.append(avg_score)
            
        if len(length_scores) > 0:
            results[length] = length_scores
            
        wrapped_env.close()
        
    pygame.quit()
    
    # Rysuj wykres
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red']
    
    x_cycles = list(range(breakthrough_cycle, cycles_total + 1, breakthrough_cycle))
    
    for i, (length, scores) in enumerate(results.items()):
        plt.plot(x_cycles[:len(scores)], scores, marker='o', label=f"Length {length}", color=colors[i % len(colors)], linewidth=2)
        
    plt.xlabel('Cumulative Training Cycles')
    plt.ylabel('Average Survival Time (Frames)')
    plt.title(f'AI Learning Progression per {breakthrough_cycle} Cycles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 520)
    plt.show()

if __name__ == "__main__":
    run_experiment()
