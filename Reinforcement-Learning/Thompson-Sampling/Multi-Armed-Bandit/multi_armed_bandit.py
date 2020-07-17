# based on a chapter "Your first AI model - Beware the bandits!" 
# of the book "AI Crash Course" Hadelin De Ponteves

import numpy as np

# every slot machine gives some chance of winning
# but AI doesn't know it

# приватными полями моделируется поведение знает/не знает другой объект об этих полях или должен ли знать или не должен

class AIEnv():
  def __init__(self, conversion_rates, n_samples):
    # допустим это приватные поля
    # самое главное поле скрыто

    self.conversion_rates = conversion_rates

    self.N = n_samples
    self.slots_number = len(self.conversion_rates)

  def get_target_dataset(self):
    X = np.zeros((self.N, self.slots_number)) # empty matrix

    for i in range(self.N):
      for j in range(self.slots_number):
        if np.random.rand() < self.conversion_rates[j]:
          X[i][j] = 1

    return X

class ReinforcementLearningAI():

  # искусственный интеллект знает между какими машинами он может выбирать
  # знает, к чему нужно стремиться

  def __init__(self, env):
    # окружение - input ИИ
    # количество слотов и идеальный датасет - это те параметры которые пришли извне
    self.slots_number = env.slots_number
    self.target_dataset = env.get_target_dataset()

    # внутренние (приватные) параметры ИИ
    self.n_pos_reward = np.zeros(self.slots_number)
    self.n_neg_reward = np.zeros(self.slots_number)

  def set_target_dataset(self):
    self.target_dataset = target_dataset

  # симулируем отдельный раунд
  def simulate_sample(self, sample_number):
    selected = self.get_decision() # output ИИ, угаданный результат

    # сравниваем с идеальным раундом
    # получаем ответ от симуляции, но окружающая среда не меняется, меняется внутреннее состояние машины
    # каждый раунд рейтинг слота либо возрастает, либо падает

    if (self.target_dataset[sample_number][selected] == 1):
      # награда возрастает
      self.n_pos_reward[selected] += 1
    else:
      self.n_neg_reward[selected] += 1

  # как именно формируется решение ИИ
  def get_decision(self):
    win_rates = [self.get_beta_distribution(slot) for slot in range(self.slots_number)]
    return win_rates.index(max(win_rates))

  def get_beta_distribution(self, slot):
    # здесь используя внутреннее состояние ИИ выбирается результат
    # beta принадлежит модулю random а не math... 
    return np.random.beta(self.n_pos_reward[slot] + 1, self.n_neg_reward[slot] + 1)

class Simulation():

  def __init__(self, aiEnv, ai):
    self.aiEnv = aiEnv
    self.ai = ai

  def get_summary(self):

    for i in range(self.aiEnv.N):
      self.ai.simulate_sample(i)

    n_selected = ai.n_pos_reward + ai.n_neg_reward
    for each in range(self.aiEnv.slots_number):
      print(f"Машина {each} была выбрана {n_selected[each]} раз")


aiEnv = AIEnv(
    conversion_rates=[0.15, 0.04, 0.13, 0.11, 0.05],
    n_samples=10000
)
ai = ReinforcementLearningAI(aiEnv)
simulation = Simulation(aiEnv, ai)

simulation.get_summary()

# It consists of choosing the action that maximizes the expected reward with respect to a randomly drawn belief.

