import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import gymnasium as gym
import heapq
from scipy.stats import truncnorm


class DynamicPricingEnv(gym.Env):
    def __init__(self):
        # Load road map
        self.road_map = ((plt.imread("map_environment.png")[:,:,0])).astype(int)
        self.road_map_agent = ((plt.imread("map_agent.png")[:,:,0])).astype(int)        
        self.Nrow, self.Ncol = np.shape(self.road_map)
        self.white_pixels = np.column_stack(np.where((self.road_map==1) & (self.road_map_agent==1)))
        self.Nwhite = len(self.white_pixels)
        self.distance_matrix = np.load('pre_computed_distance_matrix.npy')
        
        
        # Gridify the area
        self.Nrow_g, self.Ncol_g = 3, 5
        self.distance_grid_dict = None
        self.grid_dist = None
        self.dist_prob = None
        self.row_grid_indices = None
        self.col_grid_indices = None
        self.gridify()
        
        self.Horizon = 60*12 # One time = 5 seconds. 12*60 is the number of 5 seconds in an hour.
        self.MaxDrivers = 10
        self.MaxRideCost = 1 # Normalized to 1
        self.commission_ratio = 0.2
        self.max_x = 1 # Normalized to 1
        self.delta = self.max_x/self.Ncol
        self.max_y = self.Nrow*self.delta
        self.MaxDistance = self.delta*900  # Found to be 828 pixels after 500 random sample generation.
                                           # Average is about 340 pixels.
        self.AvgDistanceToPassenger = self.delta*90 # Found 90 pixels to be the average distance between driver and passenger.
        self.MaxDistanceToPassenger = self.delta*350
        
        self.MaxTheta_p = self.MaxRideCost/((1 - 0.7)*self.MaxDistance)
        self.MaxTheta_d = -self.MaxRideCost/(np.log(1-0.2)*self.MaxDistance)
        
        # Observation space
        passenger_space = gym.spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),high=np.array([self.max_x, self.max_y, self.max_x, self.max_y, self.MaxTheta_p]))
        driver_space =  gym.spaces.Sequence(gym.spaces.Box(low=np.array([0.0, 0.0, 0.0]),high=np.array([self.max_x, self.max_y, self.MaxTheta_d])))
        self.observation_space = gym.spaces.Tuple([passenger_space, driver_space])
        
        # Action space
        self.action_space = gym.spaces.Box(low=0, high=self.MaxRideCost)
        
        
        self.passenger_location = None
        self.passenger_destination = None
        self.driver_locations = None
        
        self.destination_distance = None
        self.destination_distance_estimate_passenger = None
        self.driver_distance = None
        
        self.passenger_sensitivity = None
        self.driver_sensitivity = None        
        self.passenger_sensitivity_estimate = None
        self.driver_sensitivity_estimate = None
        
        self.observation = None        
        self.t = None
        
    
    def gridify(self):
        self.distance_grid_dict = {}
        for r1, c1 in product(np.arange(self.Nrow_g), np.arange(self.Ncol_g)):
            for r2, c2 in product(np.arange(self.Nrow_g), np.arange(self.Ncol_g)):
                dist = abs(r1-r2)+abs(c1-c2)
                if dist in self.distance_grid_dict:
                    self.distance_grid_dict[dist].append((r1,c1,r2,c2))
                else:
                    self.distance_grid_dict[dist] = [(r1,c1,r2,c2)]
        
        self.grid_dist = np.array([0, 1, 2, 3, 4, 5, 6]).astype(int)
        self.dist_prob = np.array([0.1, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1])
        
        delta = int(self.Nrow/self.Nrow_g)
        remainder = self.Nrow - delta*self.Nrow_g
        self.row_grid_indices = np.zeros((self.Nrow_g,2), dtype=np.int32)
        i = 0
        for r in range(self.Nrow_g):
            if remainder>0:
                self.row_grid_indices[r,:] = (i,i+delta+1)
                remainder-=1
                i+=(delta+1)
            else:
                self.row_grid_indices[r,:] = (i,i+delta)
                i+=delta
                
        delta = int(self.Ncol/self.Ncol_g)
        remainder = self.Ncol - delta*self.Ncol_g
        self.col_grid_indices = np.zeros((self.Ncol_g,2), dtype=np.int32)
        i = 0
        for c in range(self.Ncol_g):
            if remainder>0:
                self.col_grid_indices[c,:] = (i,i+delta+1)
                remainder-=1
                i+=(delta+1)
            else:
                self.col_grid_indices[c,:] = (i,i+delta)
                i+=delta
    
    
    def step(self, action):
        assert self.observation is not None, "Call reset before using step method!"
        assert self.t<self.Horizon, "The number of time slots exceeds the time horizon!"
        
        reward = self.generate_reward(action)
        
        self.t+=1
        
        self.generate_ride_request()
        
        terminated= False
        truncated = False
        if self.t>=self.Horizon:
            truncated = True
        
        return self.observation, reward, terminated, truncated, {}
    
    
    def reset(self):
        self.t = 0
        self.generate_ride_request()
        return self.observation, {}
        
    
    def generate_reward(self, action):
        
        price = min(self.MaxRideCost, action)
        
        # Check if the passenger accepts
        prob = 1-price/(self.destination_distance_estimate_passenger*self.passenger_sensitivity)
        prob = max(0.0, prob)
        if np.random.uniform()>prob:
            return 0.0
        
        # Check if a driver accepts
        sorted_index = np.argsort(self.driver_distance)
        driver_accepted = False # False means no one accepted
        for idx in sorted_index:
            effective_price = (1-self.commission_ratio)*price
            effective_distance = self.destination_distance + self.driver_distance[idx]
            prob = 1-np.exp(-effective_price/(effective_distance*self.driver_sensitivity[idx]))
            if np.random.uniform()<=prob:
                driver_accepted = True
                break
        
        if not(driver_accepted):
            return 0.0
        else:
            return self.commission_ratio*price
        
    
    def generate_ride_request(self):
        # Passenger location and destination
        invalid = True
        while invalid:
            self.generate_passenger_location_destination()
            self.destination_distance = self.shortest_path(self.passenger_location, self.passenger_destination)
            if self.destination_distance>0:
                invalid = False        
        self.destination_distance_estimate_passenger = self.noise(self.destination_distance)
        
        # Driver location
        Ndrivers = np.random.randint(1,self.MaxDrivers+1)
        # Ndrivers = self.MaxDrivers
        self.driver_locations = self.generate_driver_locations(Ndrivers, self.passenger_location)
        Ndrivers = len(self.driver_locations)
        
        self.driver_distance = []
        for start in self.driver_locations:
            dist = self.shortest_path(start, self.passenger_location)
            self.driver_distance.append(dist)
        
        
        # Price sensitivity (both drivers and the passenger)
        self.generate_sensitivities()
        self.passenger_sensitivity_estimate = self.noise(self.passenger_sensitivity)
        self.driver_sensitivity_estimate = [self.noise(val) for val in self.driver_sensitivity]
        
        # Generate observation
        passenger_observation = np.zeros(5)
        passenger_observation[0] = self.passenger_location[1]*self.delta
        passenger_observation[1] = self.passenger_location[0]*self.delta
        passenger_observation[2] = self.passenger_destination[1]*self.delta
        passenger_observation[3] = self.passenger_destination[0]*self.delta
        passenger_observation[4] = self.passenger_sensitivity_estimate
        
        driver_observation = []
        for i in range(Ndrivers):
            temp = np.zeros(3)
            temp[0] = self.driver_locations[i][1]*self.delta
            temp[1] = self.driver_locations[i][0]*self.delta
            temp[2] = self.driver_sensitivity_estimate[i]
            driver_observation.append(temp)
        
        self.observation = (passenger_observation, tuple(driver_observation))
    
    
    def generate_sensitivities(self):
        self.passenger_sensitivity = np.random.uniform(0.05*self.MaxTheta_p, 0.95*self.MaxTheta_p)
        
        Ndrivers = len(self.driver_locations)
        self.driver_sensitivity = [np.random.uniform(0.05*self.MaxTheta_d, 0.95*self.MaxTheta_d) for i in range(Ndrivers)]
                
    
    def noise(self, true_val):
        lb = 0.9*true_val
        ub = 1.1*true_val
        mean = true_val
        stddev = 0.25*true_val
        return truncnorm.rvs(a=(lb-mean)/stddev, b=(ub-mean)/stddev, loc=mean, scale=stddev)
    
    
    def generate_passenger_location_destination(self):
        dist = np.random.choice(self.grid_dist, p=self.dist_prob)
        idx = np.random.randint(len(self.distance_grid_dict[dist]))
        
        # Passenger location
        r, c = self.distance_grid_dict[dist][idx][:2]
        min_row, max_row = self.row_grid_indices[r,:]
        min_col, max_col = self.col_grid_indices[c,:]
        region = self.road_map[min_row:max_row, min_col:max_col]
        local_white_pixels = np.column_stack(np.where(region==1))
        Nlocal = len(local_white_pixels)        
        idx2 = np.random.randint(Nlocal)
        row = local_white_pixels[idx2][0] + min_row
        col = local_white_pixels[idx2][1] + min_col
        self.passenger_location = (row, col)
        
        # Passenger destination
        r, c = self.distance_grid_dict[dist][idx][2:]
        min_row, max_row = self.row_grid_indices[r,:]
        min_col, max_col = self.col_grid_indices[c,:]
        region = self.road_map[min_row:max_row, min_col:max_col]
        local_white_pixels = np.column_stack(np.where(region==1))
        Nlocal = len(local_white_pixels)
        idx2 = np.random.randint(Nlocal)
        row = local_white_pixels[idx2][0] + min_row
        col = local_white_pixels[idx2][1] + min_col
        self.passenger_destination = (row, col)
    
        
    def generate_driver_locations(self, Ndrivers, passenger_location):
        delta_row = int(0.5*self.Nrow/self.Nrow_g)
        min_row = max(0, passenger_location[0] - delta_row)
        max_row = min(self.Nrow-1, passenger_location[0] + delta_row)
        
        delta_col = int(0.5*self.Ncol/self.Ncol_g)
        min_col = max(0, passenger_location[1] - delta_col)
        max_col = min(self.Ncol-1, passenger_location[1] + delta_col)
        
        driver_region = self.road_map[min_row:max_row+1, min_col:max_col+1]
        local_white_pixels = np.column_stack(np.where(driver_region==1))
        Nlocal = len(local_white_pixels)
        
        driver_locations_idx = []
        driver_locations = []
        for i in range(Ndrivers):
            idx = np.random.randint(Nlocal)
            if idx in driver_locations_idx:
                idx = np.random.randint(Nlocal)
            
            if idx not in driver_locations_idx:
                driver_locations_idx.append(idx)
                row_d = local_white_pixels[idx][0] + min_row
                col_d = local_white_pixels[idx][1] + min_col
                driver_locations.append(tuple([row_d, col_d]))  # (row, col)
                
        return driver_locations
    
    
    def shortest_path(self, src, dst):
        open_set = []
        heapq.heappush(open_set, (0, src))
        g_score = {src: 0}
        directions = [(-1,0), (1,0), (0,-1), (0,1)] # (d_row, d_col) AND this is for Manhattan distance      
        path_length = np.inf
        while open_set:
            _, current = heapq.heappop(open_set)

            if current==dst:
                path_length = g_score[current]
                break

            for d_row, d_col in directions:
                row, col = current[0] + d_row, current[1] + d_col
                neighbor = (row, col)
                if row>=0 and row<self.Nrow and col>=0 and col<self.Ncol and self.road_map[row,col] == 1:
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g<g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        
                        ar, ac = neighbor
                        br, bc = dst
                        h = np.max(np.abs(self.distance_matrix[:,ar,ac]-self.distance_matrix[:,br,bc]))
                        f_score = tentative_g + h
                        
                        heapq.heappush(open_set, (f_score, neighbor))

        return path_length*self.delta