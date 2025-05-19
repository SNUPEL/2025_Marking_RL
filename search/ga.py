import math
import os
import time
import random
import numpy as np
import pandas as pd
import pickle
# import torch # Not used in the provided GA snippet
from tqdm import tqdm


# from environment import NESTING # Assuming this is not needed for the GA logic itself if coords are passed in

class GeneticAlgorithmNestingSAStyle:
    def __init__(self,
                 population_size=50, generations=100,
                 mutation_rate_seq=0.02, mutation_rate_dir=0.01,
                 crossover_rate=0.9,
                 elite_size=5, tournament_size=5, initialize_method="random"):
        """
        GA for Nesting, styled after sa.py (solution representation and evaluation).
        Instance-specific data (origin, coords, etc.) is set by _initialize_instance_specifics.
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_seq = mutation_rate_seq
        self.mutation_rate_dir = mutation_rate_dir
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.initialize_method = initialize_method

        # These will be set per instance by _initialize_instance_specifics
        self.origin = None
        self.coords = None  # Normalized, (2N, 2) numpy array, where N is num_unique_parts
        self.coords_paired = None  # Normalized, (2N, 2) numpy array
        self.num_parts = 0  # Number of unique parts

        self.population = []  # List of solution arrays
        self.fitness = np.zeros(population_size)
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.current_initial_solution = None  # Stores the first solution created for the current instance
        self.current_initial_fitness = float('inf')  # Fitness of current_initial_solution

    def _process_input_data(self, data_tuple_raw):
        """
        Processes raw instance data tuple (origin, loc, loc_paired)
        and normalizes coordinates, similar to sa.py.
        Sets self.origin, self.coords, self.coords_paired, self.num_parts.
        """
        if data_tuple_raw is None or len(data_tuple_raw) < 3:
            # print("Warning in _process_input_data: data_tuple_raw is invalid.")
            self.origin, self.coords, self.coords_paired = None, np.array([]), np.array([])
            self.num_parts = 0
            return

        origin_np = np.array(data_tuple_raw[0], dtype=np.float32)
        coords_raw_np = np.array(data_tuple_raw[1], dtype=np.float32)
        coords_paired_raw_np = np.array(data_tuple_raw[2], dtype=np.float32)

        if coords_raw_np.ndim != 2 or coords_raw_np.shape[1] != 2:
            # print(f"Warning: coords_raw_np has unexpected shape: {coords_raw_np.shape}. Expected (M, 2).")
            self.origin, self.coords, self.coords_paired = origin_np, coords_raw_np, coords_paired_raw_np
            self.num_parts = 0
            return

        if coords_raw_np.size > 0:
            length_max = np.max(coords_raw_np)
            if length_max == 0:
                length_max = 1.0
            coords_normalized = coords_raw_np / length_max
            coords_paired_normalized = coords_paired_raw_np / length_max
        else:
            coords_normalized = coords_raw_np
            coords_paired_normalized = coords_paired_raw_np

        self.coords = coords_normalized
        self.coords_paired = coords_paired_normalized
        self.origin = origin_np
        self.num_parts = int(self.coords.shape[0] // 2)

        if self.coords.shape[0] % 2 != 0 and self.coords.size > 0:
            # print(
            # f"Warning: Number of rows in coords ({self.coords.shape[0]}) is odd. num_parts calculation might be incorrect.")
            pass

    def _initialize_instance_specifics(self, problem_instance_data_tuple):
        """Initializes GA for a specific instance's data."""
        self._process_input_data(problem_instance_data_tuple)

        if self.num_parts == 0 and (
                self.coords is not None and self.coords.size > 0):  # Allow num_parts=0 if coords are also empty
            # print("Warning: num_parts is 0 after processing data. GA may not function correctly.")
            # return False # Keep it true, _evaluate_solution should handle num_parts=0
            pass

        self.fitness = np.zeros(self.population_size)
        self.population = []
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.current_initial_solution = None
        self.current_initial_fitness = float('inf')
        return True

    def _evaluate_solution(self, solution_array):
        if self.coords is None or self.coords_paired is None or self.origin is None:
            return float('inf')
        if self.num_parts == 0:  # If there are no parts, cost is 0
            return 0.0
        if solution_array is None or solution_array.size == 0:  # Invalid solution
            return float('inf')

        seq_part = solution_array[:self.num_parts].astype(int)
        dir_part = solution_array[self.num_parts:].astype(int)

        try:
            actual_part_indices_in_sequence = seq_part
            actual_directions_for_sequence = dir_part
            idx = 2 * actual_part_indices_in_sequence + actual_directions_for_sequence

            if np.any(idx >= self.coords.shape[0]) or np.any(idx < 0):
                return float('inf')

            coords_A_sides = self.coords[idx]
            coords_B_sides = self.coords_paired[idx]
        except IndexError:
            return float('inf')

        starts_points_for_path = np.concatenate([coords_A_sides, self.origin[np.newaxis, ...]], axis=0)
        end_points_for_path = np.concatenate([self.origin[np.newaxis, ...], coords_B_sides], axis=0)
        cost = np.linalg.norm(end_points_for_path - starts_points_for_path, ord=2, axis=-1).sum()
        return cost

    def _create_individual_random(self):
        if self.num_parts == 0: return np.array([])
        sequence = np.random.permutation(self.num_parts)
        direction = np.random.choice(2, size=self.num_parts)
        return np.concatenate([sequence, direction])

    def _create_individual_nn(self):
        if self.num_parts == 0 or self.coords is None or self.coords.size == 0 or self.origin is None:
            # print("  [GA DEBUG _create_individual_nn] Cannot run NN: num_parts is 0 or coords/origin not set. Falling back to random.")
            return self._create_individual_random()

        coords_with_origin = np.concatenate([self.origin[np.newaxis, ...], self.coords], axis=0)

        if coords_with_origin.shape[0] <= 1:
            return self._create_individual_random()

        distance_matrix = np.linalg.norm(coords_with_origin[:, np.newaxis, :] - coords_with_origin[np.newaxis, :, :],
                                         ord=2, axis=-1)
        np.fill_diagonal(distance_matrix, np.inf)
        distance_matrix[:, 0] = np.inf

        current_node_in_dist_matrix = 0
        sequence_nn = []
        direction_nn = []
        visited_unique_parts = [False] * self.num_parts

        for _ in range(self.num_parts):
            if np.all(np.isinf(distance_matrix[current_node_in_dist_matrix])):
                break

            next_node_overall_idx = np.argmin(distance_matrix[current_node_in_dist_matrix])
            selected_coord_row_idx = next_node_overall_idx - 1
            order = selected_coord_row_idx // 2
            dir_val = selected_coord_row_idx % 2

            temp_dist_row = np.copy(distance_matrix[current_node_in_dist_matrix])
            while order < 0 or order >= self.num_parts or visited_unique_parts[order]:  # Added boundary check for order
                temp_dist_row[next_node_overall_idx] = np.inf
                if np.all(np.isinf(temp_dist_row)):
                    order = -1
                    break
                next_node_overall_idx = np.argmin(temp_dist_row)
                selected_coord_row_idx = next_node_overall_idx - 1
                order = selected_coord_row_idx // 2
                dir_val = selected_coord_row_idx % 2

            if order == -1 or order < 0 or order >= self.num_parts: break  # ensure valid order before append

            sequence_nn.append(order)
            direction_nn.append(dir_val)
            visited_unique_parts[order] = True

            distance_matrix[:, selected_coord_row_idx + 1] = np.inf
            paired_coord_row_idx = selected_coord_row_idx ^ 1
            distance_matrix[:, paired_coord_row_idx + 1] = np.inf
            current_node_in_dist_matrix = paired_coord_row_idx + 1
            if current_node_in_dist_matrix >= distance_matrix.shape[0]:
                break

        if len(sequence_nn) < self.num_parts:
            remaining_parts_indices = [p for p in range(self.num_parts) if not visited_unique_parts[p]]
            random.shuffle(remaining_parts_indices)
            sequence_nn.extend(remaining_parts_indices)
            num_dirs_needed = self.num_parts - len(direction_nn)
            if num_dirs_needed > 0:
                direction_nn.extend(np.random.choice(2, size=num_dirs_needed).tolist())

        if not sequence_nn:  # if sequence_nn is still empty (e.g. num_parts was 0 and slipped through)
            return self._create_individual_random()

        final_sequence = np.array(sequence_nn[:self.num_parts])
        final_direction = np.array(direction_nn[:self.num_parts])
        return np.concatenate([final_sequence, final_direction])

    def _initialize_population(self):
        # print("[GA DEBUG _initialize_population] Creating initial population...")
        self.population = []
        # This variable will hold the first individual, especially if it's from NN,
        # to be used as a base for mutations.
        base_individual_for_population_creation = None

        # Step 1: Create the first/base individual
        if self.initialize_method == "nearest neighbor":
            # print("  [GA DEBUG _initialize_population] Creating first individual using Nearest Neighbor.")
            first_individual = self._create_individual_nn()
            if first_individual.size > 0:
                # If NN method, this first_individual is our base for subsequent mutations.
                base_individual_for_population_creation = first_individual
        elif self.initialize_method == "random":
            # print("  [GA DEBUG _initialize_population] Creating first individual randomly.")
            first_individual = self._create_individual_random()
        else:  # Default or unknown method
            # print(f" [GA DEBUG _initialize_population] Unknown initialize_method '{self.initialize_method}'. Using random for first individual.")
            first_individual = self._create_individual_random()

        # Process the created first_individual
        if first_individual is not None and first_individual.size > 0:
            self.population.append(first_individual)
            self.current_initial_solution = np.copy(first_individual)
            self.current_initial_fitness = self._evaluate_solution(self.current_initial_solution)
            # print(f"  [GA DEBUG _initialize_population] First/Base individual created. Initial fitness: {self.current_initial_fitness:.4f}")
        else:  # First individual creation failed
            self.current_initial_solution = None
            self.current_initial_fitness = float('inf')  # Fitness is infinity if no solution
            # print("[GA DEBUG _initialize_population] Failed to create the first/base individual.")
            # Ensure base_individual_for_population_creation is None if the first attempt failed.
            base_individual_for_population_creation = None

        # Step 2: Fill the rest of the population
        # current_pop_count is 0 if first_individual failed, 1 if it succeeded.
        current_pop_count = len(self.population)

        for _ in range(current_pop_count, self.population_size):
            individual_to_add = None
            # If method is NN and a valid base NN individual was created, mutate it.
            if self.initialize_method == "nearest neighbor" and \
                    base_individual_for_population_creation is not None and \
                    base_individual_for_population_creation.size > 0:
                # print(f"  [GA DEBUG _initialize_population] Mutating NN base for a new individual.")
                mutated_individual = self._mutate(np.copy(base_individual_for_population_creation))
                if mutated_individual.size > 0:
                    individual_to_add = mutated_individual
                else:  # Mutation resulted in an invalid individual, fall back to random
                    # print(f"  [GA DEBUG _initialize_population] Mutation of NN base failed, adding random.")
                    individual_to_add = self._create_individual_random()
            else:
                # For "random" method, or if "nearest neighbor" failed to produce a base,
                # or if base_individual_for_population_creation is invalid for any reason,
                # create a new random individual.
                # print(f"  [GA DEBUG _initialize_population] Adding random individual (method: {self.initialize_method}).")
                individual_to_add = self._create_individual_random()

            if individual_to_add is not None and individual_to_add.size > 0:
                self.population.append(individual_to_add)
            # else:
            # print(f"  [GA DEBUG _initialize_population] Failed to create an additional individual, skipping.")

        if not self.population and self.population_size > 0:
            # This means even the first individual creation failed, and loop for others might not have run or also failed.
            # current_initial_fitness should already be float('inf') in this case.
            # print("  [GA DEBUG _initialize_population] CRITICAL: Population is empty after initialization and pop_size > 0.")
            pass  # current_initial_fitness is already inf
        # elif self.population:
        # print(f"[GA DEBUG _initialize_population] Total population of size {len(self.population)} created.")
        # pass

    def _evaluate_population(self):
        if not self.population:  # if population is empty
            self.best_fitness_overall = float('inf')
            # No need to update self.best_solution_overall as it should be None or from a previous run
            return

        for i, individual in enumerate(self.population):
            self.fitness[i] = self._evaluate_solution(individual)

        # Ensure fitness array is not accessed if empty, though population check above should cover.
        if self.fitness.size > 0 and len(self.population) > 0:
            # Only consider valid fitness values for finding the minimum
            valid_fitness_indices = np.where(self.fitness != float('inf'))[0]
            if valid_fitness_indices.size > 0:
                current_best_idx_pop = valid_fitness_indices[np.argmin(self.fitness[valid_fitness_indices])]
                if self.fitness[current_best_idx_pop] < self.best_fitness_overall:
                    self.best_fitness_overall = self.fitness[current_best_idx_pop]
                    self.best_solution_overall = np.copy(self.population[current_best_idx_pop])
            # If all fitnesses are inf, best_fitness_overall remains as it is (potentially inf)
        # else: # Population might be empty, or fitness array not matching
        # self.best_fitness_overall will retain its value (inf if no valid solution found yet)

    def _tournament_selection(self):
        mating_pool = []
        if not self.population or len(self.population) < self.tournament_size:  # ensure population is large enough
            # If population is too small for tournament, can return copies of existing ones or handle error
            # For now, if not enough for a full tournament, just return what we have, or rely on caller to check pool size
            return [np.copy(ind) for ind in self.population]  # Return copies to avoid direct modification issues

        for _ in range(self.population_size):  # Create a new pool of parents for next generation
            try:
                participants_indices = random.sample(range(len(self.population)), self.tournament_size)
                # Ensure fitness values are valid before finding min
                winner_idx = -1
                min_fitness_val = float('inf')
                for idx_sel in participants_indices:
                    if self.fitness[idx_sel] < min_fitness_val:
                        min_fitness_val = self.fitness[idx_sel]
                        winner_idx = idx_sel

                if winner_idx != -1:  # Found a valid winner
                    mating_pool.append(self.population[winner_idx])
                elif self.population:  # All participants had inf fitness, pick one randomly
                    mating_pool.append(self.population[random.choice(participants_indices)])

            except ValueError:  # sample larger than population
                if self.population:  # if population exists, just add a random member
                    mating_pool.append(random.choice(self.population))
                else:  # Should not happen if previous checks are in place
                    break
        return mating_pool

    def _crossover(self, parent1_sol, parent2_sol):
        # 부모 해로부터 순서(seq)와 방향(dir) 분리
        p1_seq, p1_dir = parent1_sol[:self.num_parts].astype(int), parent1_sol[self.num_parts:].astype(int)
        p2_seq, p2_dir = parent2_sol[:self.num_parts].astype(int), parent2_sol[self.num_parts:].astype(int)

        size_seq = self.num_parts

        # --- 순서 배열 교차: 위치 기반 교차 (Position-Based Crossover, POS) ---

        # 자식1 순서 배열 초기화
        c1_s = np.full(size_seq, -1, dtype=int)
        if size_seq > 0:  # 부품이 있을 경우에만 교차 수행
            # 부모1로부터 상속받을 위치의 개수 랜덤 선택 (최소 1개)
            num_pos_to_inherit_c1 = random.randint(1, size_seq)
            # 부모1로부터 상속받을 위치들 랜덤 선택
            positions_from_p1 = sorted(random.sample(range(size_seq), k=num_pos_to_inherit_c1))

            # 선택된 위치의 유전자를 부모1로부터 자식1으로 복사
            for pos in positions_from_p1:
                c1_s[pos] = p1_seq[pos]

            # 자식1의 나머지 위치를 부모2의 유전자로 채움 (순서 유지, 중복 방지)
            p2_fill_idx = 0  # 부모2에서 가져올 유전자 인덱스
            for i in range(size_seq):
                if c1_s[i] == -1:  # 자식1의 해당 위치가 아직 비어있다면
                    # 자식1에 아직 없는 유전자를 부모2에서 찾을 때까지 p2_fill_idx 증가
                    while p2_seq[p2_fill_idx] in c1_s:
                        p2_fill_idx += 1
                    c1_s[i] = p2_seq[p2_fill_idx]  # 해당 유전자로 채움
                    p2_fill_idx += 1  # 다음 부모2 유전자로 이동
        else:  # 부품이 없으면 빈 배열
            c1_s = np.array([], dtype=int)

        # 자식2 순서 배열 초기화 (자식1과 대칭적으로 수행)
        c2_s = np.full(size_seq, -1, dtype=int)
        if size_seq > 0:
            num_pos_to_inherit_c2 = random.randint(1, size_seq)
            positions_from_p2 = sorted(random.sample(range(size_seq), k=num_pos_to_inherit_c2))

            for pos in positions_from_p2:
                c2_s[pos] = p2_seq[pos]

            p1_fill_idx = 0
            for i in range(size_seq):
                if c2_s[i] == -1:
                    while p1_seq[p1_fill_idx] in c2_s:
                        p1_fill_idx += 1
                    c2_s[i] = p1_seq[p1_fill_idx]
                    p1_fill_idx += 1
        else:
            c2_s = np.array([], dtype=int)

        # --- 방향 배열 교차: 균등 교차 (기존 방식 유지) ---
        c1_d, c2_d = np.copy(p1_dir), np.copy(p2_dir)
        if self.num_parts > 0:  # 부품이 있을 경우에만 방향 교차
            for i in range(self.num_parts):
                if random.random() < 0.5:  # 50% 확률로 교환
                    c1_d[i], c2_d[i] = c2_d[i], c1_d[i]
        else:  # 부품이 없으면 빈 배열
            c1_d = np.array([], dtype=int)
            c2_d = np.array([], dtype=int)

        # 자식 해 결합
        child1 = np.concatenate([c1_s, c1_d])
        child2 = np.concatenate([c2_s, c2_d])

        return child1, child2

    def _mutate(self, individual_sol):
        if individual_sol is None or individual_sol.size == 0:
            return np.array([])  # Return empty if input is invalid
        mut_sol = np.copy(individual_sol)
        # Sequence Mutation (Swap)
        if random.random() < self.mutation_rate_seq:
            if self.num_parts >= 2:
                m_idx1, m_idx2 = random.sample(range(self.num_parts), 2)
                mut_sol[m_idx1], mut_sol[m_idx2] = mut_sol[m_idx2], mut_sol[m_idx1]
        # Direction Mutation (Flip)
        for i in range(self.num_parts):
            if random.random() < self.mutation_rate_dir:
                mut_sol[self.num_parts + i] = 1 - mut_sol[self.num_parts + i]
        return mut_sol

    def run_ga_for_instance(self, instance_data_tuple):
        if not self._initialize_instance_specifics(instance_data_tuple):
            # print("  [GA DEBUG run_ga_for_instance] Instance specifics initialization failed.")
            return None, float('inf'), float('inf'), 0.0

        start_run_time = time.time()

        # _initialize_population sets self.current_initial_fitness and self.population
        self._initialize_population()

        # Store the fitness of the very first solution created (NN or random) for reporting
        initial_objective_value_for_run = self.current_initial_fitness

        if not self.population or (self.population[0].size == 0 and self.num_parts > 0):
            # print("  [GA DEBUG run_ga_for_instance] Initial population creation failed or is invalid.")
            # If population is empty, best_fitness_overall is inf, initial_objective_value_for_run is also inf
            return None, self.best_fitness_overall, initial_objective_value_for_run, time.time() - start_run_time

        self._evaluate_population()  # Evaluate initial pop, updates self.best_fitness_overall

        # print(f"[GA DEBUG run_ga_for_instance] Initial best (pop eval): {self.best_fitness_overall:.4f} (NN/First Sol fit: {initial_objective_value_for_run:.4f})")

        for gen_idx_run_loop in range(self.generations):
            # if gen_idx_run_loop % 10 == 0 :
            # print(f"[GA DEBUG run_ga_for_instance] Generation {gen_idx_run_loop + 1}/{self.generations}, Current Best: {self.best_fitness_overall:.4f}")

            mating_pool_run_loop = self._tournament_selection()
            if not mating_pool_run_loop:  # No individuals to select from
                # print("  [GA DEBUG run_ga_for_instance] Mating pool is empty. Ending generation loop.")
                break

            new_pop_run_loop = []
            # Elitism: Carry over best individuals
            if self.fitness.size > 0 and self.elite_size > 0 and len(self.population) > 0:
                # Filter out inf fitness values before sorting for elites
                valid_fitness_indices = np.where(self.fitness != float('inf'))[0]
                if valid_fitness_indices.size > 0:
                    # Sort based on actual fitness values, then get original indices
                    sorted_valid_indices = valid_fitness_indices[np.argsort(self.fitness[valid_fitness_indices])]
                    elite_indices_run_loop = sorted_valid_indices[:min(self.elite_size, len(sorted_valid_indices))]
                    for idx_e_run_loop in elite_indices_run_loop:
                        new_pop_run_loop.append(np.copy(self.population[idx_e_run_loop]))

            needed_offspring_run_loop = self.population_size - len(new_pop_run_loop)
            offspring_list_run_loop = []

            if len(mating_pool_run_loop) >= 2 and needed_offspring_run_loop > 0:
                pool_idx_run_loop = 0
                while len(offspring_list_run_loop) < needed_offspring_run_loop:
                    p1_ga = mating_pool_run_loop[pool_idx_run_loop % len(mating_pool_run_loop)]
                    p2_ga = mating_pool_run_loop[
                        (pool_idx_run_loop + 1) % len(mating_pool_run_loop)]  # Ensure +1 is also valid
                    pool_idx_run_loop = (pool_idx_run_loop + 2)  # Ensure wrap around for next iteration

                    if random.random() < self.crossover_rate:
                        c1_ga, c2_ga = self._crossover(p1_ga, p2_ga)
                    else:
                        c1_ga, c2_ga = np.copy(p1_ga), np.copy(p2_ga)

                    offspring_list_run_loop.append(self._mutate(c1_ga))
                    if len(offspring_list_run_loop) < needed_offspring_run_loop:
                        offspring_list_run_loop.append(self._mutate(c2_ga))
            elif mating_pool_run_loop and needed_offspring_run_loop > 0:  # Not enough for pairs, or only one in pool
                while len(offspring_list_run_loop) < needed_offspring_run_loop and mating_pool_run_loop:
                    offspring_list_run_loop.append(
                        self._mutate(np.copy(random.choice(mating_pool_run_loop))))  # Mutate a random one from pool

            new_pop_run_loop.extend(offspring_list_run_loop)

            if not new_pop_run_loop and self.population_size > 0:
                # print("  [GA DEBUG run_ga_for_instance] New population is empty. Ending generation loop.")
                break  # Stop if new population is empty

            self.population = [ind for ind in new_pop_run_loop if ind.size > 0][
                              :self.population_size]  # Ensure valid individuals and cap size

            if self.population:
                self._evaluate_population()
            else:  # No valid individuals to form next generation
                # print("  [GA DEBUG run_ga_for_instance] Population became empty after selection/reproduction.")
                break

        duration_run_final_val = time.time() - start_run_time
        # print(f"  [GA DEBUG run_ga_for_instance] Instance processing complete. Final best fitness: {self.best_fitness_overall:.4f}")
        return self.best_solution_overall, self.best_fitness_overall, initial_objective_value_for_run, duration_run_final_val


# --- 메인 실행 블록 ---
if __name__ == '__main__':
    print(">>> GA 평가 시작 <<<")

    # --- 실행 설정 (여기서 직접 수정) ---
    GA_POP_SIZE = 50
    GA_GENERATIONS = 1000
    GA_MUT_RATE_SEQ = 0.1
    GA_MUT_RATE_DIR = 0.02
    GA_CROSS_RATE = 0.95
    GA_ELITE_SIZE = 4
    GA_TOURNAMENT_K_SELECT = 3

    DATA_DIRECTORY_MAIN = '../data/case1/'
    RESULTS_OUTPUT_DIR_MAIN = '../results_ga_sa_style_nesting/'

    # MODIFICATION: This variable will now be passed to the GA constructor
    GA_INITIALIZE_METHOD = "nearest neighbor"  # Options: "random", "nearest neighbor"

    INSTANCE_INDEX_RANGE_START = 0
    INSTANCE_INDEX_RANGE_END = 1000

    output_columns_main = ['dataset_file', 'instance_index', 'initial_cost_ga', 'final_cost_ga', 'time_seconds_ga']
    all_run_results_list_main = []

    if not os.path.isdir(DATA_DIRECTORY_MAIN):
        print(f"오류: 데이터 디렉토리를 찾을 수 없습니다 - {DATA_DIRECTORY_MAIN}")
        exit()

    # Check if DATA_DIRECTORY_MAIN is relative to the script or an absolute path
    # For robustness if script is not in 'search' folder:
    script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    if not os.path.isabs(DATA_DIRECTORY_MAIN):
        DATA_DIRECTORY_MAIN = os.path.join(script_dir, DATA_DIRECTORY_MAIN)
    if not os.path.isabs(RESULTS_OUTPUT_DIR_MAIN):
        RESULTS_OUTPUT_DIR_MAIN = os.path.join(script_dir, RESULTS_OUTPUT_DIR_MAIN)

    if not os.path.isdir(DATA_DIRECTORY_MAIN):
        print(f"오류: 데이터 디렉토리를 찾을 수 없습니다 (절대 경로 시도 후): {DATA_DIRECTORY_MAIN}")
        exit()

    dataset_file_list_main = [f for f in os.listdir(DATA_DIRECTORY_MAIN) if f.endswith(".pkl")]
    if not dataset_file_list_main:
        print(f"오류: 데이터 디렉토리 '{DATA_DIRECTORY_MAIN}'에 .pkl 파일이 없습니다.")
        exit()

    ga_runner_main = GeneticAlgorithmNestingSAStyle(
        population_size=GA_POP_SIZE,
        generations=GA_GENERATIONS,
        mutation_rate_seq=GA_MUT_RATE_SEQ,
        mutation_rate_dir=GA_MUT_RATE_DIR,
        crossover_rate=GA_CROSS_RATE,
        elite_size=GA_ELITE_SIZE,
        tournament_size=GA_TOURNAMENT_K_SELECT,
        initialize_method=GA_INITIALIZE_METHOD
    )
    print(f"GA Runner 객체 초기화 완료. 초기화 방법: {GA_INITIALIZE_METHOD}")

    for pkl_filename_main in dataset_file_list_main:
        print(f"\nProcessing dataset file: {pkl_filename_main}")
        current_dataset_filepath_main = os.path.join(DATA_DIRECTORY_MAIN, pkl_filename_main)
        try:
            with open(current_dataset_filepath_main, 'rb') as f_handle_main:
                raw_data_from_pkl_main = pickle.load(f_handle_main)
            print(f"  '{pkl_filename_main}' 로드 완료. 총 인스턴스 수: {len(raw_data_from_pkl_main)}")
        except Exception as e_load_main:
            print(f"  오류: '{pkl_filename_main}' 파일 로드 실패 - {e_load_main}")
            continue

        num_total_instances_main = len(raw_data_from_pkl_main)
        start_idx_main = INSTANCE_INDEX_RANGE_START
        end_idx_main = min(INSTANCE_INDEX_RANGE_END, num_total_instances_main)

        instances_to_process_this_file_main = raw_data_from_pkl_main[start_idx_main:end_idx_main]
        if not instances_to_process_this_file_main:
            print(f"  '{pkl_filename_main}'에서 지정된 범위 [{start_idx_main}-{end_idx_main - 1}]에 평가할 인스턴스가 없습니다.")
            continue

        print(
            f"  '{pkl_filename_main}'의 인스턴스 {start_idx_main}부터 {end_idx_main - 1}까지 총 {len(instances_to_process_this_file_main)}개에 대해 GA 실행...")

        for instance_idx_in_file_main, current_instance_data_tuple_main in enumerate(
                tqdm(instances_to_process_this_file_main, desc=f"GA {pkl_filename_main}")):
            actual_instance_index_main = start_idx_main + instance_idx_in_file_main
            print(f"\n  Processing instance {actual_instance_index_main} from {pkl_filename_main}")
            try:
                _best_sol_ga_res, best_cost_ga_res, initial_cost_ga_res, duration_ga_res = \
                    ga_runner_main.run_ga_for_instance(current_instance_data_tuple_main)

                print(
                    f"    Instance {actual_instance_index_main}: Initial Cost = {initial_cost_ga_res:.4f}, Final Cost = {best_cost_ga_res:.4f}, Time = {duration_ga_res:.2f}s")

                all_run_results_list_main.append([
                    pkl_filename_main, actual_instance_index_main, initial_cost_ga_res, best_cost_ga_res,
                    duration_ga_res
                ])
            except Exception as e_ga_run_loop_main:
                print(
                    f"    오류: 인스턴스 {actual_instance_index_main} ({pkl_filename_main}) GA 실행 중 문제 발생 - {e_ga_run_loop_main}")
                import traceback

                traceback.print_exc()
                all_run_results_list_main.append([
                    pkl_filename_main, actual_instance_index_main, math.inf, math.inf, 0
                ])

    if all_run_results_list_main:
        results_df_to_save_main = pd.DataFrame(all_run_results_list_main, columns=output_columns_main)
        os.makedirs(RESULTS_OUTPUT_DIR_MAIN, exist_ok=True)
        excel_filename_final_output_main = f"GA_pop{GA_POP_SIZE}_gen{GA_GENERATIONS}_init-{GA_INITIALIZE_METHOD.replace(' ', '')}_results.xlsx"
        excel_filepath_final_output_main = os.path.join(RESULTS_OUTPUT_DIR_MAIN, excel_filename_final_output_main)

        try:
            with pd.ExcelWriter(excel_filepath_final_output_main, engine='openpyxl') as writer_obj_final_main:
                results_df_to_save_main.to_excel(writer_obj_final_main, sheet_name="GA_Nesting_SA_Style", index=False)
            print(f"\n모든 결과가 다음 Excel 파일에 저장되었습니다: {excel_filepath_final_output_main}")
        except Exception as e_excel_final_main:
            print(f"최종 Excel 파일 저장 중 오류 발생: {e_excel_final_main}")
            csv_filepath_fallback_main = os.path.splitext(excel_filepath_final_output_main)[0] + ".csv"
            try:
                results_df_to_save_main.to_csv(csv_filepath_fallback_main, index=False)
                print(f"결과가 CSV 파일로 대신 저장되었습니다: {csv_filepath_fallback_main}")
            except Exception as e_csv_fallback_final_main:
                print(f"CSV 파일 저장 중에도 오류 발생: {e_csv_fallback_final_main}")
    else:
        print("\nGA 평가 결과가 없습니다.")

    print(">>> GA 평가 종료 <<<")