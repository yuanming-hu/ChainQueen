class Memo:

  def __init__(self):
    self.steps = []
    self.actuations = []
    self.initial_state = None
    self.initial_feed_dict = {}
    self.iteration_feed_dict = {}
    self.point_visualization = []
    self.vector_visualization = []
    self.stepwise_loss = None

  def update_stepwise_loss(self, step):
    if step is None:
      return
    if self.stepwise_loss is None:
      self.stepwise_loss = step
      return

    def add(a, b):
      if a is list:
        for x, y in zip(a, b):
          add(x, y)
      else:
        a += b

    add(self.stepwise_loss, step)
