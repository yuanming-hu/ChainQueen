import cv2
import os
class Export:
  def __init__(self, directory, fps = 60):
    self.cnt = 0
    self.dir = directory
    if os.path.exists(directory):
      import shutil
      shutil.rmtree(directory, ignore_errors = True)
    os.makedirs(directory)
    self.last = None
    self.fps = fps

  def __call__(self, img):
    if img.max() <= 1:
      img = img * 255
    img = img.astype('uint8')
    name = '{:0=5}.png'.format(self.cnt)
    cv2.imwrite(os.path.join(self.dir, name), img)
    self.cnt += 1
    self.last = img.copy()
  
  def wait(self, sec = 1):
    frame = int(sec * self.fps)
    for i in range(frame):
      self(self.last)

  def export(self):
    fps, d = self.fps, self.dir
    os.system('cd {}\nti video {}'.format(d, fps))
    os.system('mv {} ./{}.mp4'.format(os.path.join(d, 'video.mp4'), d))
    import shutil
    shutil.rmtree(d, ignore_errors = True)

if __name__ == '__main__':
  pass
