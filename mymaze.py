import numpy as np
from gymnasium import spaces

def crossProduct(p1, p2, p3):
    # p1p2 X p1p3
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def onSegment(p, seg):
  a, b = seg
  x, y = p
  return min(a[0], b[0]) <= x <= max(a[0], b[0]) and min(a[1], b[1]) <= y <= max(a[1], b[1])

def isSegmentIntersect(seg1, seg2):
  a, b = seg1
  c, d = seg2

  d1 = crossProduct(a, b, c)
  d2 = crossProduct(a, b, d)
  d3 = crossProduct(c, d, a)
  d4 = crossProduct(c, d, b)

  if d1 * d2 < 0 and d3 * d4 < 0:
    return True
 
  if d1 == 0 and onSegment(c, seg1): return True
  if d2 == 0 and onSegment(d, seg1): return True
  if d3 == 0 and onSegment(a, seg2): return True
  if d4 == 0 and onSegment(b, seg2): return True

  return False

import pygame

WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
RED = (255,0,0)
YELLOW = (255,255,0)
ORANGE = (255,128,0)
class myEnv(object):
    def __init__(self, render=False):
        self.size = 2
        self._start_point = np.array((0.375*self.size, 0.375*self.size))
        self._high_target_location= np.array((0.25*self.size, 0.75*self.size))
        self._low_target_location= None#np.array((0.75*self.size, 0.75*self.size))
        self._target_r = 0.125*self.size
        self._danger = [(0, 0, 0.25*self.size, 0.5*self.size)]#,(0.625*self.size, 0.625*self.size, self.size, self.size)] # (x_left,y_top,x_right,y_bottom)
        self._wall_points = {'A': (0.25*self.size,0.25*self.size), 'B': (0.25*self.size,0.5*self.size), 'C': (0.5*self.size,0.5*self.size), 'D': (0.5*self.size,0.25*self.size)}
        self.observation_space = spaces.Box(0, self.size, shape=(2,), dtype=float)
        self._timestep = 0
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=float)
        self._np_random = None
        self._seed = None
        self._agent_location = self._start_point.copy()

        # render
        self._render = render
        if render:
            pygame.init()
            self._screen_size = (400, 400)
            self._scale = self._screen_size[0] / self.size
            self._screen = pygame.display.set_mode(self._screen_size)
            pygame.display.set_caption("Monster Demo")
        else:
            self._screen_size = None
            self._scale = 0
            self._screen = None

    def render(self):
        pygame.event.get()
        screen = self._screen
        s = self._scale
        screen.fill(WHITE)
        # WALLS
        pygame.draw.lines(screen, BLACK, 0, [(x*s, y*s) for (x, y) in self._wall_points.values()], 1)
        pygame.draw.lines(screen, BLACK,1, [(0, 0),(0,self.size*s),(self.size*s, self.size*s),(self.size*s, 0)], 1)
        pygame.draw.rect(screen, BLUE, [x*s for x in self._danger[0]])
        hx, hy = self._high_target_location
        pygame.draw.circle(screen,YELLOW,(hx*s, hy*s),self._target_r*s)
        if self._low_target_location is not None:
            lx, ly = self._low_target_location
            pygame.draw.circle(screen,ORANGE,(lx*s, ly*s),self._target_r*s)
        ax, ay = self._agent_location
        pygame.draw.circle(screen,RED,(ax*s, ay*s),0.05*s)
        pygame.display.flip()

    def hit_wall(self, action):
        dx, dy = action
        x, y = self._agent_location
        vec_seg = ((x,y),(x+dx,y+dy))
        wp = self._wall_points
        wall1 = (wp['A'],wp['B'])
        wall2 = (wp['B'],wp['C'])
        wall3 = (wp['C'],wp['D'])
        return isSegmentIntersect(vec_seg,wall1) or isSegmentIntersect(vec_seg,wall2) or isSegmentIntersect(vec_seg,wall3)
    
    def in_wall(self, location):
        x, y = location
        return 0.25*self.size <= x <= 0.5*self.size and 0.25*self.size <= y <= 0.5*self.size
    
    def in_danger(self, location):
        x, y = location
        in_danger1 = self._danger[0][0] <= x <= self._danger[0][2] and self._danger[0][1] <= y <= self._danger[0][3] 
        #in_danger2 = self._danger[1][0] <= x <= self._danger[1][2] and self._danger[1][1] <= y <= self._danger[1][3] 
        return in_danger1 #or in_danger2
    
    def get_out(self, location):
        x, y = location
        return x >= self.size or x < 0 or y >= self.size or y < 0

    def win(self, location):
        get_high = np.linalg.norm(location - self._high_target_location) <= self._target_r
        if self._low_target_location is None:
            get_low = False
        else:
            get_low = np.linalg.norm(location - self._low_target_location) <= self._target_r
        assert not (get_high and get_low)
        is_win = get_low or get_high
        reward = 500*get_high + 40*get_low
        return is_win, reward

    def _get_obs(self):
        return self._agent_location

    def reset(self):

        self._agent_location = self._start_point.copy()
        self._timestep = 0
        observation = self._get_obs()
        if self._render:
            self.render()
        return observation
    
    def step(self, action):
        assert action.shape[0] == 2
        terminated = False
        reward = 0
        self._timestep += 1
        new_location = self._agent_location + action
        if self.hit_wall(action):
            terminated = True
            reward = -500
            if self.in_wall(self._agent_location):
                new_location = np.clip(new_location,np.ones(2)*0.25*self.size, np.ones(2)*0.5*self.size)
            else:
                # hit walls outside
                new_location = np.clip(new_location, np.ones(2)*0.5*self.size, np.ones(2))
        elif self.in_danger(new_location):
            terminated = True
            reward = -500
            new_location = np.clip(new_location,np.zeros(2),np.ones(2)*self.size)
        elif self.get_out(new_location):
            terminated = True
            reward = -500
            new_location = np.clip(new_location,np.zeros(2),np.ones(2)*self.size)
        else:
            terminated, reward = self.win(new_location)
            new_location = np.clip(new_location,np.zeros(2),np.ones(2)*self.size)
        
        # An episode is done iff the agent has reached the target
        self._agent_location = new_location
        observation = self._get_obs()
        if self._render:
            self.render()
        return observation, reward, terminated or self._timestep == 100

