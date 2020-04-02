import visdom
import time
import numpy as np

class Visualizer():
    def __init__(self, env='default', port=8097):
        super(Visualizer, self).__init__()
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, port=port)
        self.index = {}
        self.log_text=''

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append')
        self.index[name] = x + 1

    def img(self, name, img_):
        self.vis.image(img_.cpu().numpy(),
                       win=name,
                       opts=dict(title=name))

    def log(self, info, win='log_text'):
        self.log_text += '[{time}-{info}<br>]'.format(time=time.strftime('%Y%m%d_%H:%M:%S'), info=info)
        self.vis.text(self.log_text, win)

    def __getattr__(self, item):
        return getattr(self.vis, item)